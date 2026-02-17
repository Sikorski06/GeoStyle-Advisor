import cv2
import mediapipe as mp
import streamlit as st

# KONFIGURACJA STRONY STREAMLIT 
st.set_page_config(page_title="GeoStyle Advisor", layout="wide")
st.title("GeoStyle Advisor: Weryfikacja Systemu Wizyjnego")

# KONFIGURACJA MEDIAPIPE
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicjalizacja modelu
# max_num_faces=1: Skupiamy się na jednej osobie
# refine_landmarks=True: Zwiększa precyzję wokół oczu i ust
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# UI APLIKACJI
st.sidebar.header("Panel Sterowania")
run_camera = st.sidebar.checkbox("Uruchom Kamerę", value=False)
show_mesh = st.sidebar.checkbox("Rysuj Siatkę (Mesh)", value=True)

# Placeholder - tutaj będzie wyświetlany obraz wideo
frame_placeholder = st.empty()

# GŁÓWNA PĘTLA
if run_camera:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Błąd: Nie wykryto kamery. Sprawdź uprawnienia lub podłączenie.")
    else:
        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Nie można odczytać klatki z kamery.")
                break

            # Przetwarzanie obrazu
            # Odbicie lustrzane dla naturalnego odczucia
            frame = cv2.flip(frame, 1)
            # Konwersja BGR (OpenCV) na RGB (MediaPipe/Streamlit)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Aby przyspieszyć, oznaczamy obraz jako 'tylko do odczytu' przed przekazaniem do modelu
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Rysowanie wyników na obrazie
            if results.multi_face_landmarks and show_mesh:
                for face_landmarks in results.multi_face_landmarks:
                    # Rysowanie siatki (Tesselation)
                    mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Rysowanie konturów
                    mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

            # Wyświetlenie w Streamlit
            frame_placeholder.image(rgb_frame, channels="RGB")

        cap.release()
else:
    st.info("Zaznacz 'Uruchom Kamerę' w panelu bocznym, aby rozpocząć detekcję.")