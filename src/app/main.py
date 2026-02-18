import cv2
import mediapipe as mp
import streamlit as st
import sys
import os
import numpy as np
import time

# Konfiguracja ścieżek
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.geometry import GeometryEngine
from core.recommender import HairstyleRecommender

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="GeoStyle Advisor", page_icon="✂️", layout="centered")

# --- CSS: POPRAWA UI ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        font-size: 2.5rem;
        color: #FAFAFA;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_engines():
    return GeometryEngine(), HairstyleRecommender()

try:
    engine, recommender = init_engines()
except FileNotFoundError:
    st.error("CRITICAL ERROR: Nie znaleziono plików konfiguracyjnych. Uruchom najpierw `analyze_data.py`.")
    st.stop()

# --- STAN APLIKACJI ---
if 'scan_stage' not in st.session_state:
    st.session_state.scan_stage = "IDLE"
if 'scan_data' not in st.session_state:
    st.session_state.scan_data = {"front": None, "left": None, "right": None}
if 'final_shape' not in st.session_state:
    st.session_state.final_shape = None
if 'hold_start_time' not in st.session_state:
    st.session_state.hold_start_time = None

# --- MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_head_pose(landmarks):
    """Zwraca stosunek obrotu głowy (0.0 - 1.0)."""
    nose = landmarks[1]
    left_edge = landmarks[234]
    right_edge = landmarks[454]
    dist_total = abs(right_edge.x - left_edge.x)
    dist_nose_left = abs(nose.x - left_edge.x)
    return dist_nose_left / dist_total if dist_total != 0 else 0.5

def draw_overlay(image, text, progress=0.0):
    """Rysuje interfejs (tekst, pasek, owal pomocniczy) na wideo."""
    h, w, _ = image.shape
    
    # 1. Owal Pomocniczy (Celownik)
    # Rysujemy go na środku ekranu, żebyś wiedział gdzie ustawić głowę
    center_x, center_y = int(w / 2), int(h / 2)
    axes_x, axes_y = int(w / 3.2), int(h / 2.0) # Proporcje typowej twarzy
    
    # Rysujemy przerywany lub półprzezroczysty owal (symulacja kolorem szarym)
    cv2.ellipse(image, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, (200, 200, 200), 2)
    
    # 2. Instrukcje tekstowe
    # Czarne tło dla czytelności
    cv2.rectangle(image, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(image, text, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # 3. Pasek Postępu
    if progress > 0:
        bar_width = int(w * progress)
        bar_color = (0, int(255 * progress), 255 - int(255 * progress)) # Gradient
        cv2.rectangle(image, (0, 65), (bar_width, 70), bar_color, -1)

# --- UI: HEADER ---
st.markdown('<p class="main-header">✂️ GeoStyle 3D Scanner</p>', unsafe_allow_html=True)

# --- LOGIKA SKANERA ---
scan_container = st.empty()

# PRZYCISK STARTOWY
if st.session_state.scan_stage == "IDLE":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Ustaw twarz wewnątrz szarego owalu i kliknij Start.")
        if st.button("🚀 ROZPOCZNIJ NOWY SKAN", use_container_width=True, key="start_btn"):
            st.session_state.scan_stage = "FRONT"
            st.session_state.hold_start_time = None
            st.rerun()

# PĘTLA WIDEO
if st.session_state.scan_stage in ["FRONT", "LEFT", "RIGHT"]:
    cap = cv2.VideoCapture(0)
    stop_col1, stop_col2 = st.columns([1, 6])
    with stop_col1:
        stop_button = st.button("Anuluj", key="stop_btn")
    
    video_window = st.empty()
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        overlay_text = ""
        progress = 0.0
        is_position_correct = False
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            yaw = get_head_pose(landmarks)
            
            # --- ULEPSZONE RYSOWANIE SIATKI ---
            # 1. Rysujemy gęstą siatkę (Tesselation) cienką linią
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            # 2. Rysujemy KONTUR TWARZY (Face Oval) grubszą linią
            # To pozwoli Ci zobaczyć, gdzie model widzi Twoje czoło i brodę
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3)
            )

            # Logika Stanów
            current_stage = st.session_state.scan_stage
            
            if current_stage == "FRONT":
                if 0.45 < yaw < 0.55:
                    overlay_text = "Trzymaj prosto (3s)..."
                    is_position_correct = True
                else:
                    overlay_text = "Ustaw twarz PROSTO"
                    is_position_correct = False
                    
            elif current_stage == "LEFT":
                if yaw > 0.65:
                    overlay_text = "Trzymaj lewy profil..."
                    is_position_correct = True
                else:
                    overlay_text = "Obróć w LEWO"
                    is_position_correct = False
            
            elif current_stage == "RIGHT":
                if yaw < 0.35:
                    overlay_text = "Trzymaj prawy profil..."
                    is_position_correct = True
                else:
                    overlay_text = "Obróć w PRAWO"
                    is_position_correct = False

            # Timer
            if is_position_correct:
                if st.session_state.hold_start_time is None:
                    st.session_state.hold_start_time = time.time()
                
                elapsed = time.time() - st.session_state.hold_start_time
                progress = min(elapsed / 3.0, 1.0)
                
                if elapsed >= 3.0:
                    if current_stage == "FRONT":
                        st.session_state.scan_data["front"] = landmarks
                        st.session_state.scan_stage = "LEFT"
                    elif current_stage == "LEFT":
                        st.session_state.scan_data["left"] = landmarks
                        st.session_state.scan_stage = "RIGHT"
                    elif current_stage == "RIGHT":
                        st.session_state.scan_data["right"] = landmarks
                        st.session_state.scan_stage = "ANALYZING"
                    
                    st.session_state.hold_start_time = None
                    cap.release()
                    st.rerun()
            else:
                st.session_state.hold_start_time = None
                progress = 0.0

        else:
            overlay_text = "Nie wykryto twarzy"
            st.session_state.hold_start_time = None

        draw_overlay(frame, overlay_text, progress)
        video_window.image(frame, channels="BGR")

    cap.release()

# --- ANALIZA ---
if st.session_state.scan_stage == "ANALYZING":
    with st.spinner("Przetwarzanie geometrii 3D..."):
        time.sleep(1)
        front_landmarks = st.session_state.scan_data["front"]
        shape, metrics = engine.get_face_shape(front_landmarks)
        st.session_state.final_shape = (shape, metrics)
        st.session_state.scan_stage = "RESULT"
        st.rerun()

# --- WYNIKI ---
if st.session_state.scan_stage == "RESULT":
    shape, metrics = st.session_state.final_shape
    advice = recommender.get_advice(shape)
    
    st.balloons()
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.success(f"Kształt: **{shape}**")
        st.metric("Pewność", f"{int(metrics['match_confidence']*100)}%")
        # Placeholder grafiki
        st.image(f"https://placehold.co/400x400/262730/FAFAFA?text={shape}", caption="Wizualizacja")
        
    with col_res2:
        st.markdown("### 💇‍♀️ Rekomendacje")
        st.info(advice['description'])
        st.markdown("**✅ Polecane:**")
        for h in advice['hairstyles']:
            st.markdown(f"* {h}")
        st.error(f"**❌ Unikaj:** {advice['avoid']}")
        
    st.markdown("---")
    if st.button("🔄 Nowy Skan", use_container_width=True, key="restart_btn"):
        st.session_state.scan_stage = "IDLE"
        st.session_state.scan_data = {"front": None, "left": None, "right": None}
        st.rerun()