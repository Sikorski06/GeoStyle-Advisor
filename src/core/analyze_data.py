import pandas as pd
import json
import os

# KONFIGURACJA
INPUT_FILE = "data/processed/measurements.csv"
OUTPUT_JSON = "config/face_profiles.json"

def main():
    if not os.path.exists(INPUT_FILE):
        print("Brak pliku źródłowego wektorów: measurements.csv")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Agregacja medianowa w przestrzeni 4D
    stats = df.groupby(['category', 'gender'])[['ratio_hw', 'ratio_jf', 'ratio_fw', 'ratio_fj']].median()
    
    profiles = {}
    for (shape, gender), row in stats.iterrows():
        if shape not in profiles:
            profiles[shape] = {}
        profiles[shape][gender] = {
            "ratio_hw": row["ratio_hw"],
            "ratio_jf": row["ratio_jf"],
            "ratio_fw": row["ratio_fw"],
            "ratio_fj": row["ratio_fj"]
        }

    if not os.path.exists("config"):
        os.makedirs("config")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(profiles, f, indent=4)
    
    print(f"Baza wzorców (Face Profiles) skompilowana i zapisana: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()