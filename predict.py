import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet50 import preprocess_input


model = load_model("deepfake_detector.h5")
detector = MTCNN()

def extract_face(image_path, size=(128,128)):
    img = cv2.imread(image_path)
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
       
        return preprocess_input(face.astype("float32"))
    return None

def predict_image(image_path):
    face = extract_face(image_path)
    if face is None:
        return "No face detected"
    face = np.expand_dims(face, axis=0)  
    pred = model.predict(face, verbose=0)[0][0]
    return "Deepfake Image" if pred > 0.5 else "Real Image"

print(predict_image("test.jpg"))
