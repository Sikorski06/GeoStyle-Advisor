import cv2
import numpy as np
import os

class GenderDetector:
    def __init__(self):
        self.model_dir = os.path.join(os.getcwd(), "models", "gender")
        self.proto = os.path.join(self.model_dir, "gender_deploy.prototxt")
        self.weights = os.path.join(self.model_dir, "gender_net.caffemodel")
        self.gender_list = ['Male', 'Female']
        
        try:
            self.net = cv2.dnn.readNet(self.weights, self.proto)
            print("✅ Model płci załadowany poprawnie.")
        except Exception as e:
            print(f"⚠️ Błąd ładowania modelu CV2: {e}")
            self.net = None

    def predict_gender(self, face_image):
        if self.net is None:
            return "Unknown", 0.0
            
        try:
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), 
                                         (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            idx = preds[0].argmax()
            return self.gender_list[idx], preds[0].max()
        except Exception:
            return "Unknown", 0.0