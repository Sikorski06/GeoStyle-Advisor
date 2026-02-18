import cv2
import numpy as np
import os
import urllib.request
import sys

class GenderDetector:
    def __init__(self):
        # Ustalanie ścieżek
        self.model_dir = os.path.join(os.getcwd(), "models", "gender")
        self.proto = os.path.join(self.model_dir, "gender_deploy.prototxt")
        self.weights = os.path.join(self.model_dir, "gender_net.caffemodel")
        self.gender_list = ['Male', 'Female']
        
        # Próba przygotowania modeli
        if self._prepare_model():
            try:
                self.net = cv2.dnn.readNet(self.weights, self.proto)
                print("✅ Model płci załadowany poprawnie.")
            except Exception as e:
                print(f"⚠️ Błąd ładowania modelu CV2: {e}")
                self.net = None
        else:
            print("⚠️ DETEKTOR PŁCI NIEAKTYWNY (Brak plików).")
            self.net = None

    def _download_file(self, url, path):
        try:
            print(f"Pobieranie: {url}...")
            # User-Agent udaje przeglądarkę, żeby GitHub nas nie zablokował
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(path, 'wb') as out_file:
                    out_file.write(response.read())
            
            # Weryfikacja rozmiaru (caffemodel powinien mieć > 40MB)
            if os.path.getsize(path) < 1000:
                print(f"⚠️ Pobrany plik jest za mały (błąd pobierania). Usuwam.")
                os.remove(path)
                return False
                
            return True
        except Exception as e:
            print(f"❌ Błąd pobierania: {e}")
            return False

    def _prepare_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"Utworzono katalog: {self.model_dir}")
            
        # LINKI BEZPOŚREDNIE (Raw)
        links = {
            "proto": [
                "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
                "https://raw.githubusercontent.com/brutalk/AgeGender/master/gender_deploy.prototxt"
            ],
            "weights": [
                # Ten link działa najczęściej (bezpośredni blob z parametrem raw=true)
                "https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_net.caffemodel?raw=true",
                "https://media.githubusercontent.com/media/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel"
            ]
        }
        
        # 1. Pobieranie PROTO
        if not os.path.exists(self.proto):
            print("--- Pobieranie pliku struktury (prototxt) ---")
            success = False
            for url in links["proto"]:
                if self._download_file(url, self.proto):
                    success = True
                    break
            if not success:
                print("❌ NIE UDAŁO SIĘ POBRAĆ PLIKU PROTO.")
                return False

        # 2. Pobieranie WAG (To jest ten duży plik)
        if not os.path.exists(self.weights):
            print("--- Pobieranie pliku wag (caffemodel ~45MB) ---")
            success = False
            for url in links["weights"]:
                if self._download_file(url, self.weights):
                    success = True
                    break
            
            if not success:
                print("\n" + "!"*50)
                print("❌ AUTOMATYCZNE POBIERANIE NIE POWIODŁO SIĘ.")
                print("Musisz pobrać ten plik ręcznie:")
                print(f"LINK: {links['weights'][0]}")
                print(f"ZAPISZ JAKO: {self.weights}")
                print("!"*50 + "\n")
                return False
                
        return True

    def predict_gender(self, face_image):
        if self.net is None:
            return "Unknown", 0.0
            
        try:
            # Model oczekuje obrazu 227x227 i średnich wartości (mean subtraction)
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), 
                                         (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            idx = preds[0].argmax()
            return self.gender_list[idx], preds[0].max()
        except Exception:
            return "Unknown", 0.0