import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.decomposition import PCA

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize the FaceNet embedder
embedder = FaceNet()

# Function to write labels to a file
def write_labels_to_file(labels, filename='registered_employees.txt'):
    with open(filename, 'w') as file:
        for label in labels:
            file.write(f"{label}\n")

def load_model():
    global model_svm, labels, encoder, optimal_threshold
    with open('facenet_models/new_classifier_Jun27_759.pkl', 'rb') as f:
        model_obj = pickle.load(f)
        if len(model_obj) >= 3:
            model_svm, labels, optimal_threshold = model_obj[:3]
        else:
            model_svm, labels = model_obj
            optimal_threshold = 0.85
    
    print('Registered empolyees >>>>>>>>>>>>>>>>>>> :', labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    write_labels_to_file(labels)
    
load_model()




def predict_face(face_image):
    # Read the image
    # cv.imwrite('face_from_yolo.png', face_image)
    t_im = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
    
    # Resize and process the face image
    face_img = cv.resize(t_im, (160, 160))
    face_embedding = get_embedding(face_img)

    # Predict using the SVM model
    embedding = np.expand_dims(face_embedding, axis=0)
    # print('embedding shape ................ : ',embedding.shape)

    ypreds = model_svm.predict(embedding)
    # print('Predictions:', ypreds)  # Here we only get the index
    ypreds_probability_list = model_svm.predict_proba(embedding)
   
    # Taking index and highest probability
    probabilities = ypreds_probability_list[0]
    # print('Probabilities >>>>>>>>>>>>>>>>>>> : ', probabilities)
    
    highest_probability = max(probabilities)
    # print('Highest probability >>>>>>>>>>>>>>>>>>> :', highest_probability)

    # Assigning result probability
    result_probability = highest_probability
    
    # Inverse transform to get the predicted label
    # print('aaaaaaaaaaaaa',ypreds[0])
    result = encoder.inverse_transform(ypreds)[0]
    
    if result_probability >= optimal_threshold:
        return result, result_probability
    else:
        return "unknown", result_probability

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    # print('y hat ;;;;;;;;',yhat.shape)
    return yhat[0]
