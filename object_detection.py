"""
================================================================================
        OBJECT DETECTION – PERSISTENT CAMERA + 5s CYCLE + 3-FRAME VOTING
================================================================================
HOW TO RUN:
   py -3.14 -m pip install tf-nightly opencv-python flask scikit-learn numpy pillow requests
   py -3.14 object_detection.py
================================================================================
"""
import os
import cv2
import time
import threading
from collections import Counter
import numpy as np
import pickle
import requests
import traceback
from flask import Flask, request, jsonify, Response, render_template_string

# ------------------------- CONFIGURATION ----------------------------
ESP32_IP = "192.168.137.169"          # CHANGE THIS
ESP32_PORT = 80
CONFIDENCE_THRESHOLD = 0.75
KNOWN_DURATION = 3
UNKNOWN_DURATION = 7
COOLDOWN_SECONDS = 1
MAX_OBJECTS = 30
MAX_IMAGES_PER_OBJECT = 30
CAPTURE_INTERVAL = 0.5
WEBCAM_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SCAN_WARMUP_FRAMES = 2
SCAN_TIMEOUT_SECONDS = 10
CAMERA_INIT_DELAY = 0.1
CAMERA_FLUSH_FRAMES = 3
PREWARM_TF_ON_START = True

RESULT_DISPLAY_SECONDS = 5

# -------- ACCURACY TUNING --------
SCAN_FRAMES_FOR_VOTING = 3        # 3-frame consensus
SCAN_FRAME_GAP = 0.05             # 50ms between frames (gives slight view variation)
CENTER_CROP_RATIO = 0.80          # use center 80% of frame (object focus)
FEATURE_VERSION = 2               # bumped — old saved model was v1

JPEG_QUALITY = 70
FEED_SLEEP = 0.05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_data.pkl")
VERSION_PATH = os.path.join(MODEL_DIR, "feature_version.txt")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========================== LAZY MODEL TRAINER =========================
class ModelTrainer:
    def __init__(self):
        self.tf = None
        self.MobileNetV2 = None
        self.preprocess_input = None
        self.img_to_array = None
        self.feature_extractor = None
        self.label_encoder = None
        self.svm = None
        self.is_trained = False
        self.X_train = []
        self.y_train = []
        self._sklearn_loaded = False
        self._tf_loaded = False
        self._tf_lock = threading.Lock()
        self.class_mean = None
        self.single_class_name = None
        self.distance_threshold = 1.0
        self._load_saved_data()

    def _ensure_sklearn(self):
        if self._sklearn_loaded:
            return
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if self.svm is None:
            self.svm = SVC(kernel='linear', probability=True, C=1.0)
        self._sklearn_loaded = True

    def _ensure_tf(self):
        if self._tf_loaded:
            return
        with self._tf_lock:
            if self._tf_loaded:
                return
            try:
                import tensorflow as tf
                from tensorflow.keras.applications import MobileNetV2
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                from tensorflow.keras.preprocessing.image import img_to_array
                self.tf = tf
                self.MobileNetV2 = MobileNetV2
                self.preprocess_input = preprocess_input
                self.img_to_array = img_to_array
                self.feature_extractor = MobileNetV2(
                    weights='imagenet', include_top=False, pooling='avg',
                    input_shape=(224, 224, 3)
                )
                dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
                _ = self.feature_extractor(dummy, training=False).numpy()
                self._tf_loaded = True
                print("✅ TensorFlow / MobileNetV2 loaded & warmed up.")
            except Exception as e:
                raise RuntimeError(f"Failed to load TensorFlow: {e}")

    def prewarm_tf_async(self):
        def _worker():
            try:
                t0 = time.time()
                self._ensure_tf()
                print(f"🔥 TF pre-warm done in {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"⚠️ TF pre-warm failed: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _atomic_pickle_save(self, data, filepath):
        temp_file = filepath + ".tmp"
        try:
            with open(temp_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_file, filepath)
        except Exception as e:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass
            raise e

    def _load_saved_data(self):
        try:
            if os.path.exists(FEATURES_PATH):
                with open(FEATURES_PATH, "rb") as f:
                    self.X_train, self.y_train = pickle.load(f)
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    self.svm = pickle.load(f)
                self._sklearn_loaded = True
            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, "rb") as f:
                    self.label_encoder = pickle.load(f)

            # Feature-version check
            saved_ver = 1
            if os.path.exists(VERSION_PATH):
                try:
                    with open(VERSION_PATH, "r") as f:
                        saved_ver = int(f.read().strip())
                except Exception:
                    saved_ver = 1

            if len(self.X_train) > 0:
                self.is_trained = True
                print(f"✅ Loaded existing model with {len(self.X_train)} images (feature v{saved_ver}).")
                self._recompute_single_class_params()
                if saved_ver != FEATURE_VERSION:
                    print("⚠️ IMPORTANT: Old model was trained with OLD feature extraction.")
                    print("   Accuracy will be poor until you re-train each object.")
                    print("   Delete objects on Training page & re-capture to upgrade.")
            else:
                print("ℹ️ No existing model data found.")
        except Exception as e:
            print(f"⚠️ Error loading saved model: {e}. Starting fresh.")
            self.X_train = []
            self.y_train = []
            self.is_trained = False

    def _fit_svm_if_needed(self):
        self._ensure_sklearn()
        if not self.is_trained or len(self.X_train) == 0:
            return
        unique = set(self.y_train)
        if len(unique) > 1:
            if hasattr(self.svm, "support_vectors_"):
                return
            try:
                encoded_labels = self.label_encoder.transform(self.y_train)
                self.svm.fit(self.X_train, encoded_labels)
                print("✅ SVM re‑fitted successfully.")
            except Exception as e:
                print(f"⚠️ SVM fit failed: {e}")

    def _recompute_single_class_params(self):
        unique = set(self.y_train)
        if len(unique) == 1:
            self.single_class_name = list(unique)[0]
            self.class_mean = np.mean(self.X_train, axis=0)
            distances = [np.linalg.norm(x - self.class_mean) for x in self.X_train]
            mean_dist = np.mean(distances) if distances else 0.0
            self.distance_threshold = max(mean_dist * 2.5, 0.5)
            print(f"🔹 Single-class '{self.single_class_name}', "
                  f"mean_dist={mean_dist:.3f}, threshold={self.distance_threshold:.3f}")
        else:
            self.class_mean = None
            self.single_class_name = None

    # ============ FEATURE EXTRACTION (v2 — center crop) ============
    def extract_features(self, image):
        """
        v2: crop to center CENTER_CROP_RATIO region first → background/edges ignored.
        This makes training & prediction more object-focused and stable.
        """
        self._ensure_tf()
        h, w = image.shape[:2]
        # Center crop
        ch = int(h * CENTER_CROP_RATIO)
        cw = int(w * CENTER_CROP_RATIO)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        cropped = image[y1:y1 + ch, x1:x1 + cw]
        resized = cv2.resize(cropped, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_array = self.img_to_array(rgb)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed = self.preprocess_input(img_array)
        features = self.feature_extractor(preprocessed, training=False).numpy()
        return features[0]
    # ==============================================================

    def add_training_data(self, images, object_name):
        self._ensure_sklearn()
        self._ensure_tf()
        if len(self.y_train) > 0:
            current_objects = set(self.y_train)
            if object_name not in current_objects and len(current_objects) >= MAX_OBJECTS:
                raise ValueError(f"Max {MAX_OBJECTS} objects reached.")
        for img in images:
            if img is None or img.size == 0:
                continue
            features = self.extract_features(img)
            self.X_train.append(features)
            self.y_train.append(object_name)
        unique = set(self.y_train)
        if len(unique) > 1:
            self.label_encoder.fit(self.y_train)
            encoded_labels = self.label_encoder.transform(self.y_train)
            self.svm.fit(self.X_train, encoded_labels)
            self.is_trained = True
            self.class_mean = None
            self.single_class_name = None
        else:
            self.single_class_name = list(unique)[0]
            self.class_mean = np.mean(self.X_train, axis=0)
            self.is_trained = True
        self.save_data()
        self._recompute_single_class_params()
        return len(self.y_train)

    def predict(self, image):
        """Single-frame prediction. Returns (label, confidence, is_known)."""
        self._ensure_sklearn()
        if not self.is_trained or len(self.X_train) == 0:
            return "No Model", 0.0, False
        self._ensure_tf()
        features = self.extract_features(image)
        unique = set(self.y_train)
        if len(unique) == 1:
            if self.class_mean is None:
                return "No Model", 0.0, False
            dist = np.linalg.norm(features - self.class_mean)
            confidence = float(np.exp(-dist / self.distance_threshold))
            if confidence >= CONFIDENCE_THRESHOLD:
                return self.single_class_name, confidence, True
            else:
                return "Unknown", confidence, False
        if not hasattr(self.svm, "support_vectors_"):
            self._fit_svm_if_needed()
            if not hasattr(self.svm, "support_vectors_"):
                return "SVM not fitted", 0.0, False
        features = np.array(features).reshape(1, -1)
        probs = self.svm.predict_proba(features)[0]
        max_conf = float(np.max(probs))
        pred_encoded = int(np.argmax(probs))
        pred_name = self.label_encoder.inverse_transform([pred_encoded])[0]
        is_known = max_conf >= CONFIDENCE_THRESHOLD
        return pred_name, max_conf, bool(is_known)

    def has_training_data(self):
        return self.is_trained and len(self.X_train) > 0

    def get_all_objects(self):
        return list(set(self.y_train)) if self.y_train else []

    def delete_object(self, object_name):
        self._ensure_sklearn()
        if object_name not in self.y_train:
            return False
        new_X, new_y = [], []
        for x, y in zip(self.X_train, self.y_train):
            if y != object_name:
                new_X.append(x)
                new_y.append(y)
        self.X_train, self.y_train = new_X, new_y
        if len(self.X_train) == 0:
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder
            self.is_trained = False
            self.label_encoder = LabelEncoder()
            self.svm = SVC(kernel='linear', probability=True, C=1.0)
            self.class_mean = None
            self.single_class_name = None
        else:
            self._recompute_single_class_params()
            unique = set(self.y_train)
            if len(unique) > 1:
                self.label_encoder.fit(self.y_train)
                encoded_labels = self.label_encoder.transform(self.y_train)
                self.svm.fit(self.X_train, encoded_labels)
                self.is_trained = True
            else:
                self.single_class_name = list(unique)[0]
                self.class_mean = np.mean(self.X_train, axis=0)
                self.is_trained = True
        self.save_data()
        return True

    def save_data(self):
        self._ensure_sklearn()
        try:
            self._atomic_pickle_save((self.X_train, self.y_train), FEATURES_PATH)
            self._atomic_pickle_save(self.svm, MODEL_PATH)
            self._atomic_pickle_save(self.label_encoder, LABELS_PATH)
            with open(VERSION_PATH, "w") as f:
                f.write(str(FEATURE_VERSION))
            print(f"✅ Model/data saved (feature v{FEATURE_VERSION}).")
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            raise

# ========================== CAPTURE MANAGER =============================
class CaptureManager:
    def __init__(self, model_trainer):
        self.model = model_trainer
        self.running = False
        self.thread = None
        self.cap = None
        self.lock = threading.Lock()
        self.display_frame = None
        self.captured_images = []
        self.object_name = ""
        self.max_images = MAX_IMAGES_PER_OBJECT
        self.capture_count = 0

    def _open_camera(self):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            print("❌ Capture camera could not open")
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        return True

    def _release_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None
        print("🛑 Capture camera released")

    def start(self, object_name):
        with self.lock:
            if self.running:
                return False, "Capture already running"
            if detection_manager is not None and detection_manager.camera_on:
                print("⚠️ Stopping detection camera to free webcam for capture...")
                detection_manager.stop_camera()
            self.object_name = object_name.strip()
            if not self.object_name:
                return False, "Object name required"
            self.captured_images = []
            self.capture_count = 0
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True, f"Capturing '{self.object_name}'"

    def _loop(self):
        if not self._open_camera():
            self.running = False
            return
        last_capture = 0
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                now = time.time()
                if now - last_capture >= CAPTURE_INTERVAL:
                    self.captured_images.append(frame.copy())
                    self.capture_count += 1
                    last_capture = now
                    print(f"📸 Captured {self.capture_count}/{self.max_images}")
                    if self.capture_count >= self.max_images:
                        self.running = False
                display = frame.copy()
                cv2.putText(display, f"CAPTURE: {self.capture_count}/{self.max_images}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Object: {self.object_name}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                self.display_frame = display
                time.sleep(0.01)
        except Exception as e:
            print(f"Capture loop error: {e}")
        finally:
            self._release_camera()
            self.running = False

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self._release_camera()

    def get_status(self):
        return {
            "running": self.running,
            "count": self.capture_count,
            "max": self.max_images,
            "object": self.object_name
        }

    def get_images(self):
        return list(self.captured_images)

    def clear_images(self):
        self.captured_images = []
        self.capture_count = 0

# ========================== DETECTION MANAGER ===========================
class DetectionManager:
    def __init__(self, model_trainer):
        self.model = model_trainer
        self.camera_on = False
        self.camera_thread = None
        self.cap = None
        self.lock = threading.Lock()
        self.display_frame = None
        self.last_result = None
        self.status = "READY"
        self.scan_requested = False
        self.scan_in_progress = False
        self._shutdown = threading.Event()
        self._result_clear_timer = None
        self.last_trigger_time = 0

    def _open_camera(self):
        try:
            self.cap = cv2.VideoCapture(WEBCAM_INDEX)
            if not self.cap.isOpened():
                print("❌ Detection camera could not open")
                self.cap = None
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
            time.sleep(CAMERA_INIT_DELAY)
            for _ in range(CAMERA_FLUSH_FRAMES):
                self.cap.grab()
            return True
        except Exception as e:
            print(f"Camera open error: {e}")
            self.cap = None
            return False

    def _release_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None
        print("🛑 Detection camera released")

    def start_camera(self):
        with self.lock:
            if self.camera_on:
                return True, "Camera already on"
            if capture_manager is not None and capture_manager.running:
                return False, "Capture is running. Stop capture first."
            self._shutdown.clear()
            self.camera_on = True
            self.status = "READY"
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            return True, "Camera started"

    def stop_camera(self):
        self._shutdown.set()
        self.camera_on = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        self._release_camera()
        self.status = "STOPPED"

    def request_scan(self):
        if not self.model.has_training_data():
            return False, "No trained model available. Train an object first."
        ok, msg = self.start_camera()
        if not ok:
            return False, msg
        with self.lock:
            self._cancel_result_timer()
            self.last_result = None
            self.status = "SCANNING"
            self.scan_requested = True
        return True, "Scan started"

    def _cancel_result_timer(self):
        if self._result_clear_timer is not None:
            try:
                self._result_clear_timer.cancel()
            except Exception:
                pass
            self._result_clear_timer = None

    def _schedule_clear(self):
        self._cancel_result_timer()
        self._result_clear_timer = threading.Timer(RESULT_DISPLAY_SECONDS, self._clear_result)
        self._result_clear_timer.daemon = True
        self._result_clear_timer.start()

    def _clear_result(self):
        with self.lock:
            self.last_result = None
            if self.status == "RESULT":
                self.status = "READY"
            self._result_clear_timer = None
        print("🧹 Result cleared – ready for next scan.")

    # ============ ESP32 TRIGGER (fixed) ============
    def _trigger_esp32(self, is_known, label):
        now = time.time()
        if now - self.last_trigger_time < COOLDOWN_SECONDS:
            print(f"⏭ ESP32 trigger skipped (cooldown {COOLDOWN_SECONDS}s not elapsed)")
            return
        self.last_trigger_time = now
        action = "green" if is_known else "red"
        duration = KNOWN_DURATION if is_known else UNKNOWN_DURATION
        url = f"http://{ESP32_IP}:{ESP32_PORT}/{action}?duration={duration}&t={int(now*1000)}"
        try:
            r = requests.get(url, timeout=1.5)
            print(f"✅ ESP32 -> {action} ({duration}s) | HTTP {r.status_code} | {url}")
        except Exception as e:
            print(f"❌ ESP32 unreachable: {e} | url={url}")
    # ==============================================

    # ============ 3-FRAME VOTING ============
    def _vote_predictions(self, preds):
        """
        preds: list of (label, confidence, is_known)
        Rules:
          • Group known predictions by label.
          • If any label has >= max(2, need) votes → winner (avg conf of its votes).
          • Otherwise → Unknown (avg conf of all).
        """
        if not preds:
            return "Unknown", 0.0, False

        n = len(preds)
        need = 2 if n >= 2 else 1   # 2/3, 2/2, or single-frame fallback
        known = [(p[0], p[1]) for p in preds if p[2]]

        if known:
            counter = Counter(lbl for lbl, _ in known)
            top_label, top_count = counter.most_common(1)[0]
            if top_count >= need:
                confs = [c for lbl, c in known if lbl == top_label]
                avg = sum(confs) / len(confs)
                return top_label, float(avg), True

        avg_all = sum(p[1] for p in preds) / n
        return "Unknown", float(avg_all), False
    # ========================================

    def _collect_voting_frames(self, first_frame):
        """
        Grab SCAN_FRAMES_FOR_VOTING frames total (first_frame + extras).
        Small gap gives slight camera variation → better consensus signal.
        """
        frames = [first_frame]
        for _ in range(SCAN_FRAMES_FOR_VOTING - 1):
            t0 = time.time()
            ret, f = self.cap.read()
            if ret:
                frames.append(f)
            elapsed = time.time() - t0
            if elapsed < SCAN_FRAME_GAP:
                time.sleep(SCAN_FRAME_GAP - elapsed)
        return frames

    def _camera_loop(self):
        if not self._open_camera():
            self.camera_on = False
            self.status = "CAMERA ERROR"
            self.last_result = {"label": "Camera could not open",
                                "is_known": False, "confidence": 0.0}
            return

        frame_count = 0
        try:
            while not self._shutdown.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue
                frame_count += 1

                if self.scan_requested and not self.scan_in_progress:
                    self.scan_in_progress = True
                    self.scan_requested = False
                    try:
                        t0 = time.time()
                        # 1) Grab frames for voting
                        vote_frames = self._collect_voting_frames(frame)
                        # 2) Predict each
                        preds = []
                        for f in vote_frames:
                            try:
                                p = self.model.predict(f)
                                preds.append((p[0], float(p[1]), bool(p[2])))
                            except Exception as pe:
                                print(f"Frame predict error: {pe}")
                        # 3) Vote
                        final_label, final_conf, final_known = self._vote_predictions(preds)
                        dt = time.time() - t0

                        print(f"🗳️ Votes: {preds}")
                        print(f"⚡ Final in {dt:.2f}s -> "
                              f"{'✓' if final_known else '✗'} {final_label} ({final_conf*100:.1f}%)")

                        result = {
                            "label": final_label if final_known else "Unknown",
                            "is_known": final_known,
                            "confidence": final_conf
                        }
                        self._trigger_esp32(final_known, final_label)
                        with self.lock:
                            self.last_result = result
                            self.status = "RESULT"
                            self._schedule_clear()
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        with self.lock:
                            self.status = "ERROR"
                            self.last_result = {"label": f"Prediction error: {str(e)}",
                                                "is_known": False, "confidence": 0.0}
                            self._schedule_clear()
                    finally:
                        self.scan_in_progress = False

                display = frame.copy()
                st = self.status
                if st == "SCANNING" or self.scan_in_progress:
                    cv2.putText(display, "SCANNING (3-frame vote)...", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif st == "RESULT" and self.last_result:
                    label = self.last_result.get("label", "")
                    conf = self.last_result.get("confidence", 0.0)
                    known = self.last_result.get("is_known", False)
                    color = (0, 220, 0) if known else (0, 0, 255)
                    cv2.putText(display, f"{label} ({conf*100:.1f}%)", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                elif st == "CAMERA ERROR":
                    cv2.putText(display, "CAMERA ERROR", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(display, "READY - Press SCAN", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                self.display_frame = display
                time.sleep(0.01)
        except Exception as e:
            print(f"Camera loop error: {e}")
        finally:
            self._release_camera()
            self.camera_on = False
            print("✅ Detection camera loop exited")

    def get_status(self):
        status = {
            "running": bool(self.camera_on),
            "status": str(self.status),
            "result": None
        }
        if self.last_result:
            status["result"] = {
                "label": str(self.last_result.get("label", "Unknown")),
                "is_known": bool(self.last_result.get("is_known", False)),
                "confidence": float(self.last_result.get("confidence", 0.0))
            }
        return status

# ========================== FLASK APP ================================
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

_model = None
capture_manager = None
detection_manager = None

def get_model():
    global _model
    if _model is None:
        _model = ModelTrainer()
    return _model

def get_capture_manager():
    global capture_manager
    if capture_manager is None:
        capture_manager = CaptureManager(get_model())
    return capture_manager

def get_detection_manager():
    global detection_manager
    if detection_manager is None:
        detection_manager = DetectionManager(get_model())
    return detection_manager

def init_managers():
    get_model()
    get_capture_manager()
    get_detection_manager()
    if PREWARM_TF_ON_START:
        get_model().prewarm_tf_async()

init_managers()

# ========================== HTML TEMPLATES ===========================
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trainer - Capture</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f172a; color: #e2e8f0; margin:0; padding:20px; }
        .container { max-width:1400px; margin:auto; }
        h1 { color:#38bdf8; border-bottom:2px solid #334155; padding-bottom:10px; }
        .grid { display:grid; grid-template-columns:2fr 1fr; gap:20px; margin-top:20px; }
        .video-section { background:#1e293b; border-radius:12px; padding:20px; position:relative; }
        .video-section img { width:100%; border-radius:8px; border:2px solid #334155; }
        .controls { display:flex; flex-wrap:wrap; gap:10px; margin-top:15px; align-items:center; }
        .controls input, .controls button { padding:10px 16px; border-radius:8px; border:none; font-weight:600; cursor:pointer; }
        .controls input { background:#334155; color:white; flex:1; min-width:150px; }
        .controls input::placeholder { color:#94a3b8; }
        .btn-primary { background:#2563eb; color:white; }
        .btn-success { background:#16a34a; color:white; }
        .btn-danger { background:#dc2626; color:white; }
        .btn-warning { background:#ca8a04; color:white; }
        .btn-secondary { background:#64748b; color:white; }
        button:disabled { opacity:0.5; cursor:not-allowed; }
        .info-panel { background:#1e293b; border-radius:12px; padding:20px; }
        .badge { background:#334155; padding:4px 12px; border-radius:20px; font-size:0.9em; display:inline-block; margin-right:10px; }
        .object-list { max-height:300px; overflow-y:auto; margin-top:10px; }
        .object-item { display:flex; justify-content:space-between; padding:6px 8px; background:#0f172a; border-radius:6px; margin-bottom:4px; }
        .object-item button { background:#dc2626; color:white; border:none; padding:2px 12px; border-radius:4px; cursor:pointer; }
        .object-item button:hover { background:#b91c1c; }
        #statusMessage { margin-top:10px; padding:10px; border-radius:6px; display:none; white-space:pre-wrap; word-break:break-word; max-height:200px; overflow-y:auto; }
        .msg-success { background:#064e3b; color:#a7f3d0; display:block; }
        .msg-error { background:#7f1d1d; color:#fca5a5; display:block; }
        .msg-info { background:#1e3a8a; color:#93c5fd; display:block; }
        .flex-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
        @media (max-width:800px) { .grid { grid-template-columns:1fr; } }
    </style>
</head>
<body>
<div class="container">
    <h1>📸 Training & Capture</h1>
    <p style="color:#94a3b8;">Train up to <strong>30 objects</strong> with <strong>max 30 images</strong> each.</p>
    <div class="grid">
        <div class="video-section">
            <img id="videoFeed" src="{{ url_for('capture_feed') }}" alt="Webcam">
            <div class="controls">
                <input type="text" id="objName" placeholder="Object name (e.g. Bottle)" />
                <button class="btn-primary" id="btnStartCapture">📸 Start Capture</button>
                <button class="btn-warning" id="btnStopCapture" disabled>⏹ Stop</button>
                <button class="btn-success" id="btnTrain" disabled>🚀 Train</button>
            </div>
            <div class="flex-row">
                <span class="badge" id="captureCounter">Captured: 0 / 30</span>
                <span class="badge" id="objectCounter">Objects: 0 / 30</span>
            </div>
            <div id="statusMessage"></div>
        </div>
        <div class="info-panel">
            <h3>📋 Trained Objects</h3>
            <div class="object-list" id="objectList"><p style="color:#64748b;">None yet.</p></div>
            <hr style="border-color:#334155;">
            <h3>⚙️ ESP32</h3>
            <div style="display:flex; gap:8px; margin-bottom:10px;">
                <span>IP: <span id="esp32IpDisplay">192.168.1.100</span></span>
            </div>
            <div style="display:flex; gap:8px;">
                <input type="text" id="esp32IpInput" placeholder="New IP" value="192.168.1.100" style="flex:1; padding:8px; border-radius:6px; border:none; background:#334155; color:white;">
                <button class="btn-secondary" id="btnSetIp">Update</button>
            </div>
            <hr style="border-color:#334155;">
            <div style="margin-top:10px;">
                <a href="/detect" target="_blank" style="color:#60a5fa;">🔍 Go to Detection Page</a>
            </div>
            <hr style="border-color:#334155;">
            <div style="margin-top:10px;">
                <button class="btn-secondary" id="btnTestTF">🧪 Test TensorFlow</button>
            </div>
        </div>
    </div>
</div>
<script>
    const objName = document.getElementById('objName');
    const btnStart = document.getElementById('btnStartCapture');
    const btnStop = document.getElementById('btnStopCapture');
    const btnTrain = document.getElementById('btnTrain');
    const captureCounter = document.getElementById('captureCounter');
    const objectCounter = document.getElementById('objectCounter');
    const objectList = document.getElementById('objectList');
    const statusMsg = document.getElementById('statusMessage');
    const esp32IpDisplay = document.getElementById('esp32IpDisplay');
    const esp32IpInput = document.getElementById('esp32IpInput');
    const btnSetIp = document.getElementById('btnSetIp');
    const btnTestTF = document.getElementById('btnTestTF');
    let captureInterval = null;

    function showStatus(msg, type='info', details=''){
        statusMsg.className = 'msg-'+type;
        statusMsg.style.display = 'block';
        if(details) {
            statusMsg.innerHTML = msg + '<br><span style="font-size:0.8em; color:#aaa;">' + details + '</span>';
        } else {
            statusMsg.textContent = msg;
        }
        setTimeout(() => {
            statusMsg.style.display = 'none';
            statusMsg.innerHTML = '';
        }, 10000);
    }

    async function fetchObjects(){
        try {
            const res = await fetch('/get_objects');
            const data = await res.json();
            objectCounter.textContent = `Objects: ${data.count} / ${data.max}`;
            if(data.objects.length === 0) {
                objectList.innerHTML = '<p style="color:#64748b;">None yet.</p>';
            } else {
                let html = '';
                data.objects.forEach(o => {
                    html += `<div class="object-item"><span>📦 ${o}</span><button onclick="deleteObject('${o}')">Delete</button></div>`;
                });
                objectList.innerHTML = html;
            }
        } catch(e) { console.log(e); }
    }

    window.deleteObject = async function(name){
        if(!confirm(`Delete "${name}"?`)) return;
        try {
            const res = await fetch('/delete_object', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({name})
            });
            const data = await res.json();
            showStatus(data.message, data.status);
            fetchObjects();
        } catch(e){ showStatus('Error','error'); }
    };

    async function pollStatus(){
        try {
            const res = await fetch('/capture_status');
            const data = await res.json();
            captureCounter.textContent = `Captured: ${data.count} / ${data.max}`;
            if(data.running){
                btnStart.disabled = true;
                btnStop.disabled = false;
                btnTrain.disabled = true;
            } else {
                btnStart.disabled = false;
                btnStop.disabled = true;
                btnTrain.disabled = !(data.count > 0);
                if(data.count >= data.max) {
                    showStatus(`Auto-capture done! ${data.count} images. Click Train.`, 'info');
                }
            }
        } catch(e){}
    }

    btnStart.addEventListener('click', async () => {
        const name = objName.value.trim();
        if(!name){ showStatus('Enter object name.','error'); return; }
        try {
            const res = await fetch('/start_capture', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({object_name: name})
            });
            const data = await res.json();
            if(data.success) {
                showStatus(data.message,'success');
                if(captureInterval) clearInterval(captureInterval);
                captureInterval = setInterval(pollStatus, 500);
                pollStatus();
            } else {
                showStatus(data.message,'error');
            }
        } catch(e){ showStatus('Error','error'); }
    });

    btnStop.addEventListener('click', async () => {
        await fetch('/stop_capture', {method:'POST'});
        showStatus('Stopped manually.','info');
        if(captureInterval) clearInterval(captureInterval);
        captureInterval = setInterval(pollStatus, 2000);
        pollStatus();
    });

    btnTrain.addEventListener('click', async () => {
        const name = objName.value.trim();
        if(!name){ showStatus('Name missing.','error'); return; }
        btnTrain.disabled = true;
        btnTrain.textContent = '⏳ Training...';
        try {
            const res = await fetch('/train', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({object_name: name})
            });
            const data = await res.json();
            if(data.success) {
                showStatus(data.message, 'success');
                objName.value = '';
                btnTrain.textContent = '🚀 Train';
                btnTrain.disabled = true;
                captureCounter.textContent = 'Captured: 0 / 30';
                if(captureInterval) clearInterval(captureInterval);
                captureInterval = setInterval(pollStatus, 2000);
                pollStatus();
                fetchObjects();
            } else {
                let errorMsg = '❌ ' + data.message;
                let details = data.traceback ? data.traceback : '';
                showStatus(errorMsg, 'error', details);
                btnTrain.disabled = false;
                btnTrain.textContent = '🚀 Train';
                if(details) console.error(details);
            }
        } catch(e){
            showStatus('Train failed. Check console.', 'error', e.toString());
            console.error(e);
            btnTrain.disabled = false;
            btnTrain.textContent = '🚀 Train';
        }
    });

    btnSetIp.addEventListener('click', async () => {
        const ip = esp32IpInput.value.trim();
        if(!ip){ showStatus('Enter IP.','error'); return; }
        try {
            const res = await fetch('/set_esp32_ip', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ip})
            });
            const data = await res.json();
            if(data.status === 'success') {
                esp32IpDisplay.textContent = ip;
                showStatus(data.message,'success');
            } else {
                showStatus(data.message,'error');
            }
        } catch(e){ showStatus('Error','error'); }
    });

    btnTestTF.addEventListener('click', async () => {
        const res = await fetch('/test_tf');
        const data = await res.json();
        showStatus(data.message, data.status);
    });

    fetchObjects();
    setInterval(fetchObjects, 5000);
    setInterval(pollStatus, 2000);
</script>
</body>
</html>
"""

DETECT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Detection - 3-Frame Voting</title>
    <style>
        body { margin:0; background:#0f172a; display:flex; justify-content:center; align-items:center; min-height:100vh; font-family:'Segoe UI',sans-serif; }
        .container { text-align:center; position:relative; max-width:900px; padding:20px; }
        h1 { color:#38bdf8; }
        .video-wrapper { border:4px solid #334155; border-radius:16px; overflow:hidden; box-shadow:0 0 30px rgba(56,189,248,0.2); position:relative; }
        .video-wrapper img { width:100%; display:block; }
        .status-bar { margin-top:15px; display:flex; align-items:center; justify-content:center; gap:20px; flex-wrap:wrap; }
        .status-text { color: #94a3b8; font-size:1.2em; }
        .result-text { font-weight:bold; font-size:1.4em; min-width:150px; }
        .btn-scan { background:#16a34a; color:white; border:none; padding:12px 30px; border-radius:30px; font-size:1.2em; font-weight:bold; cursor:pointer; transition:0.3s; }
        .btn-scan:disabled { opacity:0.5; cursor:not-allowed; background:#64748b; }
        .btn-scan:hover:not(:disabled) { background:#15803d; }
        .nav-link { color:#60a5fa; text-decoration:none; margin-top:20px; display:inline-block; }
        .small-note { color:#64748b; margin-top:10px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 One‑Shot Detection (3-Frame Voting)</h1>
    <p style="color:#94a3b8;">Camera stays live. Press SCAN → result shown for 5s → ready again.</p>
    <div class="video-wrapper">
        <img id="videoFeed" src="{{ url_for('detection_feed') }}" alt="Live Feed">
    </div>
    <div class="status-bar">
        <span class="status-text">Status:</span>
        <span class="result-text" id="resultDisplay">Ready</span>
    </div>
    <div style="margin-top:15px;">
        <button class="btn-scan" id="scanButton">🔍 SCAN</button>
    </div>
    <div id="scanStatus" style="margin-top:10px; color:#94a3b8; min-height:30px;">Press SCAN to start</div>
    <a href="/" class="nav-link">← Back to Training</a>
    <div class="small-note">Camera does NOT close between scans – next scan is instant.</div>
</div>
<script>
    const scanBtn = document.getElementById('scanButton');
    const resultDisplay = document.getElementById('resultDisplay');
    const scanStatus = document.getElementById('scanStatus');

    let scanRunning = false;
    let resultShownAt = 0;
    const RESULT_HOLD_MS = 5000;

    function resetUI() {
        resultDisplay.innerText = 'Ready';
        resultDisplay.style.color = '#94a3b8';
        scanStatus.innerText = 'Press SCAN to start';
        scanBtn.disabled = false;
        scanBtn.innerText = '🔍 SCAN';
        scanRunning = false;
    }

    async function startScan() {
        if (scanRunning) return;
        scanRunning = true;
        scanBtn.disabled = true;
        scanBtn.innerText = '⏳ SCANNING...';
        scanStatus.innerText = 'Scanning (3 frames)...';
        resultDisplay.innerText = 'Scanning...';
        resultDisplay.style.color = '#fbbf24';
        resultShownAt = 0;

        try {
            const response = await fetch('/scan', { method: 'POST' });
            const data = await response.json();
            if (!data.success) {
                scanStatus.innerText = 'Error: ' + data.message;
                resultDisplay.innerText = 'Error';
                resultDisplay.style.color = '#f87171';
                scanRunning = false;
                setTimeout(resetUI, RESULT_HOLD_MS);
            }
        } catch (error) {
            console.error(error);
            scanStatus.innerText = 'Request failed';
            resultDisplay.innerText = 'Error';
            resultDisplay.style.color = '#f87171';
            scanRunning = false;
            setTimeout(resetUI, RESULT_HOLD_MS);
        }
    }

    async function pollStatus() {
        try {
            const res = await fetch('/detection_status');
            const data = await res.json();

            if (data.status === 'SCANNING') {
                scanStatus.innerText = 'Scanning (3 frames)...';
                resultDisplay.innerText = 'Scanning...';
                resultDisplay.style.color = '#fbbf24';
                scanBtn.disabled = true;
                scanBtn.innerText = '⏳ SCANNING...';
                scanRunning = true;

            } else if (data.status === 'RESULT' && data.result) {
                const r = data.result;
                const label = r.label || 'Unknown';
                const conf = r.confidence ? (r.confidence * 100).toFixed(1) : '0';
                const isKnown = r.is_known;

                if (resultShownAt === 0) resultShownAt = Date.now();
                resultDisplay.innerText = label + ' (' + conf + '%)';
                resultDisplay.style.color = isKnown ? '#22c55e' : '#ef4444';
                const remain = Math.max(1, Math.ceil((RESULT_HOLD_MS - (Date.now() - resultShownAt))/1000));
                scanStatus.innerText = 'Result shown. Next scan ready in ~' + remain + 's';
                scanBtn.disabled = true;
                scanBtn.innerText = '⏳ WAIT ' + remain + 's';

            } else if (data.status === 'CAMERA ERROR' || data.status === 'ERROR') {
                let msg = data.status;
                if (data.result && data.result.label) msg = data.result.label;
                resultDisplay.innerText = msg;
                resultDisplay.style.color = '#f87171';
                scanStatus.innerText = 'Error: ' + msg;
                scanRunning = false;
                scanBtn.disabled = true;
                scanBtn.innerText = '⏳ WAIT...';
                setTimeout(resetUI, RESULT_HOLD_MS);

            } else {
                if (resultShownAt !== 0) {
                    if (Date.now() - resultShownAt >= RESULT_HOLD_MS) {
                        resultShownAt = 0;
                        resetUI();
                    }
                } else if (!scanRunning) {
                    resetUI();
                }
            }
        } catch (e) { console.log(e); }
    }

    scanBtn.addEventListener('click', startScan);
    setInterval(pollStatus, 250);
    pollStatus();
</script>
</body>
</html>
"""

# ========================== FLASK ROUTES =============================
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/detect')
def detect():
    return render_template_string(DETECT_HTML)

@app.route('/ping')
def ping():
    return "Server is alive!"

@app.route('/test_tf')
def test_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        model = MobileNetV2(weights='imagenet', include_top=False)
        return jsonify({'status': 'success',
                        'message': 'TensorFlow loaded successfully! Version: ' + tf.__version__})
    except Exception as e:
        return jsonify({'status': 'error',
                        'message': f'TensorFlow load failed: {str(e)}'}), 500

@app.route('/capture_feed')
def capture_feed():
    def generate():
        while True:
            frame = capture_manager.display_frame if capture_manager else None
            if frame is None:
                blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, "CAPTURE CAMERA OFF", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = blank
            ret, buffer = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes() + b'\r\n')
            time.sleep(FEED_SLEEP)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    try:
        data = request.get_json() or {}
        name = data.get('object_name', '').strip()
        if not name:
            return jsonify({'success': False, 'message': 'Object name required.'})
        cm = get_capture_manager()
        success, msg = cm.start(name)
        return jsonify({'success': success, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    try:
        cm = get_capture_manager()
        cm.stop()
        return jsonify({'success': True, 'message': 'Capture stopped.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/capture_status')
def capture_status():
    try:
        cm = get_capture_manager()
        return jsonify(cm.get_status())
    except Exception as e:
        return jsonify({'running': False, 'count': 0,
                        'max': MAX_IMAGES_PER_OBJECT, 'object': ''})

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json() or {}
        name = data.get('object_name', '').strip()
        if not name:
            return jsonify({'success': False, 'message': 'Object name required.'})
        cm = get_capture_manager()
        images = cm.get_images()
        if not images:
            return jsonify({'success': False,
                            'message': 'No captured images. Please capture first.'})
        model = get_model()
        total = model.add_training_data(images, name)
        cm.clear_images()
        return jsonify({
            'success': True,
            'message': f'"{name}" trained with {len(images)} images. Total images: {total}',
            'total_objects': len(model.get_all_objects())
        })
    except Exception as e:
        error_details = traceback.format_exc()
        print("Training error:\n", error_details)
        return jsonify({
            'success': False,
            'message': str(e),
            'traceback': error_details
        }), 500

@app.route('/detection_feed')
def detection_feed():
    def generate():
        while True:
            frame = detection_manager.display_frame if detection_manager else None
            if frame is None:
                blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, "PRESS SCAN", (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = blank
            ret, buffer = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes() + b'\r\n')
            time.sleep(FEED_SLEEP)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan', methods=['POST'])
def scan():
    try:
        dm = get_detection_manager()
        success, msg = dm.request_scan()
        return jsonify({'success': success, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/start_detector', methods=['POST'])
def start_detector():
    return scan()

@app.route('/stop_detector', methods=['POST'])
def stop_detector():
    try:
        dm = get_detection_manager()
        dm.stop_camera()
        return jsonify({'success': True, 'message': 'Detection camera stopped.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/detection_status')
def detection_status():
    try:
        dm = get_detection_manager()
        return jsonify(dm.get_status())
    except Exception as e:
        return jsonify({
            'running': False,
            'status': 'ERROR',
            'result': {'label': f'Status error: {str(e)}',
                       'is_known': False, 'confidence': 0.0}
        })

@app.route('/get_objects')
def get_objects():
    try:
        model = get_model()
        objs = model.get_all_objects()
        return jsonify({'objects': objs, 'count': len(objs), 'max': MAX_OBJECTS})
    except Exception as e:
        print(f"Get objects error: {e}")
        return jsonify({'objects': [], 'count': 0, 'max': MAX_OBJECTS})

@app.route('/delete_object', methods=['POST'])
def delete_object():
    try:
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        if not name:
            return jsonify({'status': 'error', 'message': 'Name required.'}), 400
        model = get_model()
        if model.delete_object(name):
            return jsonify({'status': 'success', 'message': f'"{name}" deleted.'})
        return jsonify({'status': 'error', 'message': 'Not found.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/set_esp32_ip', methods=['POST'])
def set_esp32_ip():
    global ESP32_IP
    try:
        data = request.get_json() or {}
        ip = data.get('ip', '').strip()
        if not ip:
            return jsonify({'status': 'error', 'message': 'IP required.'}), 400
        ESP32_IP = ip
        return jsonify({'status': 'success', 'message': f'ESP32 IP set to {ip}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ========================== MAIN ====================================
if __name__ == '__main__':
    print("===================================")
    print("FLASK SERVER STARTING - 3-FRAME VOTING BUILD")
    print("TRAINING: http://127.0.0.1:5000")
    print("DETECTION: http://127.0.0.1:5000/detect")
    print("===================================")
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )