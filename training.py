"""
training.py — Training pipeline.

Flow:
  1. Download dataset from GCS  (``dataset/{emp_id}/``)
  2. Augment images, save augmented to **both** local + GCS
  3. Extract FaceNet embeddings
  4. Train SVM with **emp_id** labels
  5. Find optimal threshold via ROC curve
  6. Save model locally + upload to GCS
"""
import os
import pickle

import cv2 as cv
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from config import DATASET_DIR, MODEL_PATH, MODEL_DIR


# ═══════════════════════════════════════════════════════════
# Face loading
# ═══════════════════════════════════════════════════════════
class FACELOADING:
    """Load and pre-process face images from ``dataset/{emp_id}/``."""

    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img)
        if not faces:
            # Already a tight face crop — use as-is
            return cv.resize(img, self.target_size)
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y + h, x:x + w]
        return cv.resize(face, self.target_size)

    def load_faces(self, dir_path):
        FACES = []
        for im_name in os.listdir(dir_path):
            if not im_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                path = os.path.join(dir_path, im_name)
                FACES.append(self.extract_face(path))
            except Exception:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in sorted(os.listdir(self.directory)):
            path = os.path.join(self.directory, sub_dir)
            if not os.path.isdir(path):
                continue
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded [{sub_dir}]: {len(labels)} images")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)


def get_embedding(face_img, embedder):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]


# ═══════════════════════════════════════════════════════════
# Training entry-point
# ═══════════════════════════════════════════════════════════
def train_model(
    dataset_dir=None,
    model_output_path=None,
    status_callback=None,
    use_augmentation: bool = True,
    sync_gcs: bool = True,
):
    """
    Full training pipeline.

    Returns dict with ``train_acc``, ``test_acc``, ``classes``, ``threshold``.
    """
    dataset_dir = dataset_dir or DATASET_DIR
    model_output_path = model_output_path or MODEL_PATH
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # ── 1. Download dataset from GCS ──────────────────────
    if sync_gcs:
        try:
            from gcs_storage import download_all_dataset
            log("Downloading dataset from GCS …")
            dl = download_all_dataset(dataset_dir)
            total = sum(len(v) for v in dl.values())
            log(f"Downloaded {total} files for {len(dl)} employees from GCS.")
        except Exception as e:
            log(f"⚠️  GCS download failed ({e}). Using local dataset.")

    # ── 2. Augmentation ───────────────────────────────────
    if use_augmentation:
        try:
            from augmentation import augment_dataset
            log("Running augmentation pipeline …")
            aug = augment_dataset(
                dataset_dir=dataset_dir,
                n_variants=20,
                status_callback=status_callback,
            )
            log(
                f"Augmentation done: {aug['total_source']} source → "
                f"{aug['total_generated']} variants."
            )

            # Upload augmented images to GCS
            if sync_gcs:
                try:
                    from gcs_storage import upload_dataset_folder
                    log("Uploading augmented images to GCS …")
                    for eid in sorted(os.listdir(dataset_dir)):
                        edir = os.path.join(dataset_dir, eid)
                        if os.path.isdir(edir):
                            uploaded = upload_dataset_folder(eid, edir)
                            log(f"  [{eid}] Uploaded {len(uploaded)} files to GCS.")
                except Exception as e:
                    log(f"⚠️  GCS upload of augmented images failed: {e}")

        except Exception as exc:
            log(f"⚠️  Augmentation skipped ({exc}).")

    # ── 3. Load faces ─────────────────────────────────────
    log(f"Loading faces from: {dataset_dir}")
    fl = FACELOADING(dataset_dir)
    X, Y = fl.load_classes()

    if len(X) == 0:
        raise ValueError("No face images found. Add images first.")

    log(f"Loaded {len(X)} images across {len(set(Y))} emp_ids.")

    # ── 4. Embeddings ─────────────────────────────────────
    log("Loading FaceNet embedder …")
    embedder = FaceNet()

    log("Extracting embeddings …")
    EMBEDDED_X = np.asarray([get_embedding(img, embedder) for img in X])

    npz_path = os.path.join(MODEL_DIR, "embeddings_cache.npz")
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez_compressed(npz_path, EMBEDDED_X, Y)
    log("Embeddings cached locally.")

    # ── 5. Train SVM ──────────────────────────────────────
    enc = LabelEncoder()
    enc.fit(Y)
    Y_enc = enc.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        EMBEDDED_X, Y_enc, shuffle=True, random_state=17,
    )

    log("Training SVM classifier …")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, Y_train)

    train_acc = accuracy_score(Y_train, svm.predict(X_train))
    test_acc = accuracy_score(Y_test, svm.predict(X_test))
    log(f"Train acc: {train_acc:.4f}  |  Test acc: {test_acc:.4f}")

    # ── 6. ROC threshold ──────────────────────────────────
    probs = svm.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == Y_test).astype(int)

    if len(np.unique(correct)) <= 1:
        probs = svm.predict_proba(X_train)
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == Y_train).astype(int)

    if len(np.unique(correct)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(correct, max_probs)
        idx_best = int(np.argmax(tpr - fpr))
        optimal_threshold = float(thresholds[idx_best])
        log(f"ROC optimal threshold: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.85
        log(f"Perfect classification — defaulting threshold to {optimal_threshold}")

    # ── 7. Save model ─────────────────────────────────────
    with open(model_output_path, 'wb') as f:
        pickle.dump((svm, list(enc.classes_), optimal_threshold), f)
    log(f"Model saved: {model_output_path}")

    # ── 8. Upload to GCS (best-effort) ─────────────────────
    gcs_success = False
    if sync_gcs:
        try:
            from gcs_storage import upload_model, upload_npz
            url = upload_model(model_output_path)
            log(f"✅ Model uploaded to GCS: {url}")
            url2 = upload_npz(npz_path)
            log(f"✅ Embeddings uploaded to GCS: {url2}")
            gcs_success = True
        except Exception as e:
            log(f"⚠️  GCS upload failed: {e}")
            log("Model saved locally — Pi can still fetch it from this laptop via WiFi.")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'classes': list(enc.classes_),
        'threshold': optimal_threshold,
        'gcs_success': gcs_success,
    }


if __name__ == "__main__":
    result = train_model()
    print("Training complete.")
    print(f"  Registered emp_ids: {result['classes']}")
    print(f"  Threshold: {result['threshold']:.4f}")