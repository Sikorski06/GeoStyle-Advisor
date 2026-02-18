import os
import cv2
import mediapipe as mp
import glob
import pandas as pd
import numpy as np
from gender_detector import GenderDetector

# KONFIGURACJA
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "measurements.csv")
EXTENSIONS = ["jpg", "jpeg", "png", "webp"]

# INICJALIZACJA
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
detector = GenderDetector()

def calculate_distance(p1, p2):
    # oblicza dystans euklidesowy
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(RAW_DATA_DIR, "**", f"*.{ext}"), recursive=True))
    
    print(f"Znaleziono {len(files)} plików. Rozpoczynam przetwarzanie z detekcją płci...")
    
    dataset = []

    for idx, file_path in enumerate(files):
        category = os.path.basename(os.path.dirname(file_path))
        image = cv2.imread(file_path)
        if image is None: continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # detekcja płci na klatce
            gender, _ = detector.predict_gender(image)

            # pomiary
            ratio_hw = calculate_distance(landmarks[10], landmarks[152]) / calculate_distance(landmarks[234], landmarks[454])
            ratio_jf = calculate_distance(landmarks[58], landmarks[288]) / calculate_distance(landmarks[234], landmarks[454])

            dataset.append({
                "category": category,
                "gender": gender,
                "ratio_hw": ratio_hw,
                "ratio_jf": ratio_jf
            })

        if (idx + 1) % 100 == 0:
            print(f"Przetworzono: {idx + 1}/{len(files)}")

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Zapisano dane do {OUTPUT_FILE}")

if __name__ == "__main__":
    main()