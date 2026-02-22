import numpy as np
import json
import os

CONFIG_PATH = "config/face_profiles.json"

class GeometryEngine:
    def __init__(self):
        self.profiles = self._load_profiles()

    def _load_profiles(self):
        if not os.path.exists(CONFIG_PATH):
            return {}
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def calculate_pixel_distance(self, p1, p2, width, height):
        # Ekstrakcja dystansu w rzutowaniu na absolutne piksele klatki
        dx = (p1.x - p2.x) * width
        dy = (p1.y - p2.y) * height
        return np.sqrt(dx**2 + dy**2)

    def get_face_shape(self, landmarks, gender="Female", **kwargs):
        mapped_gender = "Female" if gender in ["Kobieta", "Female"] else "Male"

        # Pobranie wymiarów ze strumienia wideo zdefiniowanych w kwargs, domyślnie 640x480 w przypadku awarii I/O
        fw = kwargs.get("frame_width", 640)
        fh = kwargs.get("frame_height", 480)

        top = landmarks[10]
        chin = landmarks[152]
        cheek_left = landmarks[234]
        cheek_right = landmarks[454]
        jaw_left = landmarks[58]
        jaw_right = landmarks[288]
        fh_left = landmarks[71]
        fh_right = landmarks[301]

        # Generacja dystansów euklidesowych z poprawioną propagacją szerokości/wysokości
        f_height = self.calculate_pixel_distance(top, chin, fw, fh)
        f_width = self.calculate_pixel_distance(cheek_left, cheek_right, fw, fh)
        j_width = self.calculate_pixel_distance(jaw_left, jaw_right, fw, fh)
        fw_width = self.calculate_pixel_distance(fh_left, fh_right, fw, fh)

        if f_width == 0: f_width = 0.001
        if j_width == 0: j_width = 0.001

        # Wektory obiektu
        current_hw = f_height / f_width
        current_jf = j_width / f_width
        current_fw = fw_width / f_width
        current_fj = fw_width / j_width

        best_match = "Unknown"
        min_distance = float("inf")
        
        # Klasyfikator Euklidesowy
        for shape, gender_data in self.profiles.items():
            metrics = gender_data.get(mapped_gender) or (list(gender_data.values())[0] if gender_data else None)
            if not metrics: continue

            t_hw = metrics.get("ratio_hw", current_hw)
            t_jf = metrics.get("ratio_jf", current_jf)
            t_fw = metrics.get("ratio_fw", current_fw)
            t_fj = metrics.get("ratio_fj", current_fj)

            # Ważony dystans euklidesowy
            dist = np.sqrt(
                2.0 * (t_hw - current_hw)**2 +
                1.5 * (t_jf - current_jf)**2 +
                1.0 * (t_fw - current_fw)**2 +
                1.0 * (t_fj - current_fj)**2
            )

            if dist < min_distance:
                min_distance = dist
                best_match = shape

        # Normalizacja odległości - zredukowany mnożnik kary 1.5 zamiast 3.0 w celu wyrównania czułości algorytmu
        confidence_raw = 1.0 - (min_distance * 1.5)
        match_confidence = max(0.01, min(0.99, confidence_raw))

        return best_match, {
            "ratio_hw": round(current_hw, 2),
            "ratio_jf": round(current_jf, 2),
            "match_confidence": round(match_confidence, 2)
        }