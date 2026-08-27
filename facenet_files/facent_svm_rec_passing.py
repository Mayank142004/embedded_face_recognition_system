"""
facent_svm_rec_passing.py — Face recognition inference module (TFLite Optimized).

Returns (emp_id, confidence) for a given face crop.
The SVM classifier is trained with emp_id as the label.
"""
import os
import cv2 as cv
import numpy as np
import pickle
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODEL_PATH, DEFAULT_CONFIDENCE_THRESHOLD, PROJECT_ROOT

# ── FaceNet TFLite Embedder (loaded once) ───────────────────
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "facenet_models", "facenet.tflite")

interpreter = None
input_details = None
output_details = None

if os.path.exists(TFLITE_MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    print(f"⚠️  TFLite FaceNet model not found at {TFLITE_MODEL_PATH}. Embeddings will fail.")

# ── Module-level globals ──────────────────────────────────
model_svm = None
labels = None
encoder = None
optimal_threshold = DEFAULT_CONFIDENCE_THRESHOLD

# ── Model loading ─────────────────────────────────────────
def load_model(model_path: str = None):
    global model_svm, labels, encoder, optimal_threshold

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

    print(f"Model loaded.  Labels (emp_ids): {labels}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(labels)

def refresh_model_from_gcs():
    try:
        from gcs_storage import download_model
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        download_model(MODEL_PATH)
        load_model()
        print("Model refreshed from GCS.")
    except Exception as e:
        print(f"GCS model refresh failed ({e}). Using local model.")

# ── Inference ─────────────────────────────────────────────
def _normalize(image):
    """Normalize using the exact method from keras-facenet (per-image mean/std)"""
    image = np.float32(image)
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
    return (image - mean) / std_adj

def get_embedding(face_img: np.ndarray) -> np.ndarray:
    """Extract 512-d FaceNet embedding using TFLite."""
    if interpreter is None:
        raise RuntimeError("TFLite FaceNet model not loaded.")
        
    if face_img.shape[:2] != (160, 160):
        face_img = cv.resize(face_img, (160, 160))
        
    processed_img = _normalize(face_img)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def predict_face(face_image: np.ndarray) -> tuple:
    if model_svm is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    t_im = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
    face_embedding = get_embedding(t_im)

    embedding = np.expand_dims(face_embedding, axis=0)

    ypreds = model_svm.predict(embedding)
    probs = model_svm.predict_proba(embedding)[0]
    highest = float(max(probs))

    result = encoder.inverse_transform(ypreds)[0]

    if highest >= optimal_threshold:
        return str(result), highest
    return "unknown", highest

# ── Initial model load ────────────────────────────────────
try:
    load_model()
except Exception:
    pass
