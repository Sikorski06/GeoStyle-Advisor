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

def calculate_distance(p1, p2, width, height):
    # Dystans euklidesowy z kompensacją proporcji pikselowych
    dx = (p1.x - p2.x) * width
    dy = (p1.y - p2.y) * height
    return np.sqrt(dx**2 + dy**2)

def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(RAW_DATA_DIR, "**", f"*.{ext}"), recursive=True))
    
    print(f"Znaleziono {len(files)} plików. Uruchamianie silnika ekstrakcji wektorów...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    detector = GenderDetector()
    
    dataset = []

    for idx, file_path in enumerate(files):
        category = os.path.basename(os.path.dirname(file_path))
        image = cv2.imread(file_path)
        if image is None: 
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape

            gender, _ = detector.predict_gender(image)

            # Ekspansja wskaźników geometrycznych
            f_height = calculate_distance(landmarks[10], landmarks[152], w, h)
            f_width = calculate_distance(landmarks[234], landmarks[454], w, h)
            j_width = calculate_distance(landmarks[58], landmarks[288], w, h)
            fw_width = calculate_distance(landmarks[71], landmarks[301], w, h) # Szerokość czoła

            # Zapobieganie błędom dzielenia przez zero (ZeroDivisionError Guard)
            if f_width == 0: f_width = 0.001
            if j_width == 0: j_width = 0.001

            # Wektory proporcji
            ratio_hw = f_height / f_width
            ratio_jf = j_width / f_width
            ratio_fw = fw_width / f_width
            ratio_fj = fw_width / j_width

            dataset.append({
                "category": category,
                "gender": gender,
                "ratio_hw": ratio_hw,
                "ratio_jf": ratio_jf,
                "ratio_fw": ratio_fw,
                "ratio_fj": ratio_fj
            })

        if (idx + 1) % 50 == 0:
            print(f"Przetworzono: {idx + 1}/{len(files)}")

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"ZAKOŃCZONO. Wektory zapisano do {OUTPUT_FILE}")

if __name__ == "__main__":
    main()