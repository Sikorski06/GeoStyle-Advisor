import numpy as np
import json
import os

CONFIG_PATH = "config/face_profiles.json"

class GeometryEngine:
    def __init__(self):
        self.profiles = self._load_profiles()

    def _load_profiles(self):
        if not os.path.exists(CONFIG_PATH):
            # Fallback jeśli brak pliku - zapobiega crashowi przy starcie
            return {}
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def get_face_shape(self, landmarks, gender="Female"):
        """
        Klasyfikuje kształt biorąc pod uwagę płeć.
        Argument gender jest teraz wymagany przez main.py.
        """
        # 1. Punkty kluczowe (MediaPipe Indices)
        top = landmarks[10]
        chin = landmarks[152]
        cheek_left = landmarks[234]
        cheek_right = landmarks[454]
        jaw_left = landmarks[58]
        jaw_right = landmarks[288]

        # 2. Obliczenia metryk
        height = self.calculate_distance(top, chin)
        width = self.calculate_distance(cheek_left, cheek_right)
        jaw_width = self.calculate_distance(jaw_left, jaw_right)

        # Zabezpieczenie przed dzieleniem przez zero
        if width == 0: width = 0.001

        ratio_hw = height / width
        ratio_jf = jaw_width / width

        # 3. Logika Decyzyjna
        best_match = "Unknown"
        min_error = float("inf")

        # Iterujemy przez profile z pliku JSON
        # Oczekiwana struktura JSON: { "Square": { "Male": {...}, "Female": {...} } }
        for shape, gender_data in self.profiles.items():
            
            # Wybór danych dla odpowiedniej płci
            if gender in gender_data:
                metrics = gender_data[gender]
            else:
                # Jeśli brak danych dla konkretnej płci, bierzemy pierwsze dostępne (fallback)
                if isinstance(gender_data, dict) and len(gender_data) > 0:
                    metrics = list(gender_data.values())[0]
                else:
                    continue

            # Obliczanie błędu dopasowania
            try:
                target_hw = metrics.get("ratio_hw", 0)
                target_jf = metrics.get("ratio_jf", 0)
                
                # Błąd euklidesowy ważony
                error = abs(target_hw - ratio_hw) + abs(target_jf - ratio_jf)
                
                if error < min_error:
                    min_error = error
                    best_match = shape
            except (KeyError, TypeError):
                continue

        # Obliczanie pewności (0-1)
        confidence = 1.0 - min(min_error, 1.0)

        return best_match, {
            "ratio_hw": round(ratio_hw, 2),
            "ratio_jf": round(ratio_jf, 2),
            "match_confidence": round(confidence, 2)
        }