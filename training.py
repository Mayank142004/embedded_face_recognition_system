import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mtcnn.mtcnn import MTCNN
detector = MTCNN()


# automate the preproseccing
class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()


    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr


    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')


from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle


def get_embedding(face_img, embedder):
    face_img = face_img.astype('float32')  # 3d (160,160)
    face_img = np.expand_dims(face_img, axis=0)  # 4D (Nonex160,160,3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512d image (1x1x512)


def train_model(
    dataset_dir='facenet_files/dataset2',
    model_output_path='facenet_models/new_classifier_Jun27_759.pkl',
    status_callback=None,
    use_augmentation: bool = True,
):
    """
    Loads all face images from dataset_dir, extracts FaceNet embeddings,
    trains an SVM classifier, and saves it to model_output_path.

    Args:
        dataset_dir:         Path to the dataset directory (one subfolder per person).
        model_output_path:   Path to save the trained SVM .pkl file.
        status_callback:     Optional callable(str) for progress reporting (e.g. st.write).
        use_augmentation:    If True (default), synthetically expand the dataset using
                             augmentation.py before extracting embeddings.  Each real
                             image produces ``n_variants=20`` lighting/pose variants
                             so the classifier generalises better with fewer real photos.

    Returns:
        dict with 'train_acc', 'test_acc', and 'classes' keys.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # ── Optional: expand dataset with augmented variants before training ──────
    if use_augmentation:
        try:
            from augmentation import augment_dataset
            log("Running image augmentation pipeline …")
            aug_result = augment_dataset(
                dataset_dir=dataset_dir,
                n_variants=20,
                status_callback=status_callback,
            )
            log(f"Augmentation done: {aug_result['total_source']} source images → "
                f"{aug_result['total_generated']} variants generated.")
        except Exception as aug_exc:
            log(f"⚠️  Augmentation skipped ({aug_exc}). Continuing with original images.")

    log(f"Loading faces from: {dataset_dir}")
    faceloading = FACELOADING(dataset_dir)
    X, Y = faceloading.load_classes()

    if len(X) == 0:
        raise ValueError("No face images found in the dataset directory. Please add images first.")

    log(f"Loaded {len(X)} face images across {len(set(Y))} classes.")

    log("Loading FaceNet embedder...")
    embedder = FaceNet()

    log("Extracting embeddings...")
    EMBEDDED_X = [get_embedding(img, embedder) for img in X]
    EMBEDDED_X = np.asarray(EMBEDDED_X)

    np.savez_compressed('faces_embeddings_done_for_officeMysr.npz', EMBEDDED_X, Y)
    log("Embeddings saved.")

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y_encoded = encoder.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        EMBEDDED_X, Y_encoded, shuffle=True, random_state=17
    )

    log("Training SVM classifier...")
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, Y_train)

    train_acc = accuracy_score(Y_train, svm_model.predict(X_train))
    test_acc = accuracy_score(Y_test, svm_model.predict(X_test))
    log(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")

    # --- Find Threshold using ROC curve ---
    probs = svm_model.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == Y_test).astype(int)
    
    if len(np.unique(correct)) <= 1:
        # Fallback to train set if test set is perfectly classified
        probs = svm_model.predict_proba(X_train)
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == Y_train).astype(int)

    if len(np.unique(correct)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(correct, max_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        log(f"Calculated optimal threshold from ROC curve: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.85
        log(f"No misclassifications found to compute ROC, defaulting threshold to {optimal_threshold}")

    with open(model_output_path, 'wb') as f:
        pickle.dump((svm_model, list(encoder.classes_), optimal_threshold), f)
    log(f"Model saved to: {model_output_path}")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'classes': list(encoder.classes_),
    }


if __name__ == "__main__":
    result = train_model()
    print("Training complete. Registered employees:", result['classes'])