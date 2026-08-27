"""
facent_svm_rec_passing_lite.py — Ultra-lightweight Face Recognition for Raspberry Pi

Uses:
- MediaPipe BlazeFace for Face Detection (No PyTorch)
- TensorFlow Lite for FaceNet Embeddings (No TensorFlow)
- Scikit-learn for SVM Classification
"""
import os
import cv2 as cv
import numpy as np
import pickle
import mediapipe as mp
import tflite_runtime.interpreter as tflite

from config import MODEL_PATH, DEFAULT_CONFIDENCE_THRESHOLD, PROJECT_ROOT

# ── FaceNet TFLite Embedder ─────────────────────────────────
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "facenet_models", "facenet.tflite")

if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"⚠️  TFLite model not found at {TFLITE_MODEL_PATH}. Embeddings will fail.")

interpreter = None
input_details = None
output_details = None

if os.path.exists(TFLITE_MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def _normalize(image):
    """Normalize using the exact method from keras-facenet (per-image mean/std)"""
    image = np.float32(image)
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
    return (image - mean) / std_adj

def get_embedding(face_img: np.ndarray) -> np.ndarray:
    """Extract 512-d FaceNet embedding from a 160×160 RGB face image using TFLite."""
    if interpreter is None:
        raise RuntimeError("TFLite FaceNet model not loaded.")
        
    # Resize just in case
    if face_img.shape[:2] != (160, 160):
        face_img = cv.resize(face_img, (160, 160))
        
    processed_img = _normalize(face_img)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ── MediaPipe Face Detector ─────────────────────────────────
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0, # 0 for short-range (up to 2m) - perfect for attendance
    min_detection_confidence=0.6
)

# ── Module-level globals (SVM) ──────────────────────────────
model_svm = None
labels = None
optimal_threshold = DEFAULT_CONFIDENCE_THRESHOLD

def load_model(model_path: str = None):
    """Load or reload the SVM classifier from a local pickle file."""
    global model_svm, labels, optimal_threshold

    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, 'rb') as f:
        model_obj = pickle.load(f)
        if len(model_obj) >= 3:
            model_svm, labels, optimal_threshold = model_obj[:3]
        else:
            model_svm, labels = model_obj
            optimal_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    print(f"Lite Model loaded. Labels: {labels}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")

def predict_face(face_image: np.ndarray) -> tuple:
    """
    Predict the employee ID from a BGR face crop.
    Returns: (emp_id, confidence)
    """
    if model_svm is None:
        raise RuntimeError("SVM Model not loaded.")

    t_im = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
    face_embedding = get_embedding(t_im)
    embedding = np.expand_dims(face_embedding, axis=0)

    ypreds = model_svm.predict(embedding)
    probs = model_svm.predict_proba(embedding)[0]
    highest = float(max(probs))

    result = labels[ypreds[0]]

    if highest >= optimal_threshold:
        return str(result), highest
    return "unknown", highest

# Try loading SVM on import
try:
    load_model()
except Exception:
    pass
