import os
import csv
from datetime import datetime

LOG_DIR = "data/logs"
LOG_FILE = os.path.join(LOG_DIR, "feedback_loop.csv")

class FeedbackLogger:
    def __init__(self):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        
        # Inicjalizacja nagłówków jeśli plik nie istnieje
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "predicted_shape", 
                    "gender", 
                    "match_confidence",
                    "ratio_hw", 
                    "ratio_jf", 
                    "ratio_fw", 
                    "ratio_fj", 
                    "is_correct"
                ])

    def log_result(self, shape, gender, metrics, is_correct):
        """Zapisuje wektor cech oraz ocenę użytkownika do pliku CSV."""
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                shape,
                gender,
                metrics.get("match_confidence", 0),
                metrics.get("ratio_hw", 0),
                metrics.get("ratio_jf", 0),
                metrics.get("ratio_fw", 0),
                metrics.get("ratio_fj", 0),
                1 if is_correct else 0
            ])