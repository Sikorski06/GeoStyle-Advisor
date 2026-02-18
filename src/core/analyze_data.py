import pandas as pd
import json
import os

# KONFIGURACJA
INPUT_FILE = "data/processed/measurements.csv"
OUTPUT_JSON = "config/face_profiles.json"

def main():
    if not os.path.exists(INPUT_FILE):
        print("Brak pliku measurements.csv")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # obliczanie median z grupowaniem po kształcie i płci
    # tworzy strukturę: kształt -> płeć -> wartości
    stats = df.groupby(['category', 'gender'])[['ratio_hw', 'ratio_jf']].median()
    
    # transformacja do słownika zagnieżdżonego
    profiles = {}
    for (shape, gender), row in stats.iterrows():
        if shape not in profiles:
            profiles[shape] = {}
        profiles[shape][gender] = {
            "ratio_hw": row["ratio_hw"],
            "ratio_jf": row["ratio_jf"]
        }

    if not os.path.exists("config"):
        os.makedirs("config")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(profiles, f, indent=4)
    
    print(f"Zapisano profile uwzględniające płeć do {OUTPUT_JSON}")

if __name__ == "__main__":
    main()