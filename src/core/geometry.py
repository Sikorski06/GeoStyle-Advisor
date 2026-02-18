import numpy as np
import json
import os

# Ścieżka do pliku konfiguracyjnego 
CONFIG_PATH = "config/face_profiles.json"

class GeometryEngine:
    def __init__(self):
        self.profiles = self._load_profiles()

    def _load_profiles(self):
        # Wczytuje ustalone progi (thresholds) z pliku JSON
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Brak pliku konfiguracyjnego: {CONFIG_PATH}. Uruchom analyze_data.py.")
        
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def calculate_distance(self, p1, p2):
        # Oblicza dystans euklidesowy między punktami (znormalizowany
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def get_face_shape(self, landmarks):
        """
        Główna funkcja klasyfikująca.
        Przyjmuje: listę punktów landmarks z MediaPipe.
        Zwraca: (kształt_twarzy, słownik_metryk).
        """
        # Pobranie kluczowych punktów (indeksy MediaPipe)
        # Wysokość
        top = landmarks[10]
        chin = landmarks[152]
        
        # Szerokość twarzy (policzki)
        cheek_left = landmarks[234]
        cheek_right = landmarks[454]
        
        # Szerokość szczęki
        jaw_left = landmarks[58]
        jaw_right = landmarks[288]

        # Obliczenie wymiarów
        height = self.calculate_distance(top, chin)
        width = self.calculate_distance(cheek_left, cheek_right)
        jaw_width = self.calculate_distance(jaw_left, jaw_right)

        # Wyliczenie wskaźników (Ratios)
        ratio_hw = height / width
        ratio_jf = jaw_width / width

        # Logika Decyzyjna (Algorytm Najbliższego Sąsiada / Regułowy)
        # Porównujemy aktualną twarz do profili z JSON i szukamy najmniejszego błędu.
        
        best_match = "Unknown"
        min_error = float("inf")

        for shape, metrics in self.profiles.items():
            # Błąd to różnica między idealnym ratio z JSON a aktualnym ratio z kamery
            # Ważymy oba wskaźniki (można dostroić wagi)
            error = abs(metrics["ratio_hw"] - ratio_hw) + abs(metrics["ratio_jf"] - ratio_jf)
            
            if error < min_error:
                min_error = error
                best_match = shape

        return best_match, {
            "ratio_hw": round(ratio_hw, 2),
            "ratio_jf": round(ratio_jf, 2),
            "match_confidence": round(1.0 / (1.0 + min_error), 2) # pewność dopasowania
        }