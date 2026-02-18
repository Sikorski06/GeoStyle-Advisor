import sys
import traceback
import os

print(f"Python Version: {sys.version}")

try:
    print("\n--- KROK 1: Import podstawowy ---")
    import mediapipe
    print(f"Lokalizacja pliku: {mediapipe.__file__}")

    print("\n--- KROK 2: Wymuszenie importu wewnętrznego (Hard Import) ---")

    # To jest linia, która zazwyczaj wywala błąd DLL, jeśli czegoś brakuje
    import mediapipe.python.solutions as solutions_module
    print("Udało się zaimportować 'mediapipe.python.solutions' bezpośrednio.")
 
    print("\n--- KROK 3: Sprawdzenie mp.solutions ---")
    import mediapipe as mp
    
    # Ręczna naprawa (Monkey Patch), jeśli standardowy import zgubił referencję
    if not hasattr(mp, 'solutions'):
        print("mp.solutions nie istnieje. Próbuję przypisać ręcznie...")
        mp.solutions = solutions_module
    
    if hasattr(mp, 'solutions'):
        print("mp.solutions jest teraz dostępne.")
    else:
        raise AttributeError("Nadal brak mp.solutions mimo przypisania.")

    print("\n--- KROK 4: Test Modelu ---")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    print("🎉 SUKCES FINALNY: Model FaceMesh utworzony!")

except ImportError:
    print("\n❌ BŁĄD KRYTYCZNY (ImportError):")
    print("System nie może załadować bibliotek DLL. Sprawdź poniższy log:")
    traceback.print_exc()
except AttributeError:
    print("\n❌ BŁĄD STRUKTURY (AttributeError):")
    traceback.print_exc()
except Exception:
    print("\n❌ NIEOCZEKIWANY BŁĄD:")
    traceback.print_exc()