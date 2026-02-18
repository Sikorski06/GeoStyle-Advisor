import cv2
import mediapipe as mp
import streamlit as st
import sys
import os
import numpy as np
import time

sys.path.append(os.path.join(os.getcwd(), 'src'))
from core.geometry import GeometryEngine
from core.recommender import HairstyleRecommender

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="GeoStyle Pro", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# --- STYLIZACJA (GLASSMORPHISM CSS) ---
st.markdown("""
<style>
    /* Główne tło */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(10, 10, 10) 90%);
        color: #ffffff;
    }
    
    /* Nagłówek */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }

    /* Karty (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Wyniki */
    .shape-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .big-metric {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.2;
    }
    
    /* Fryzury */
    .hair-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 20px;
        border-radius: 12px;
        margin: 5px;
        display: inline-block;
        transition: all 0.3s ease;
    }
    .hair-pill:hover {
        background: rgba(255, 75, 75, 0.2);
        border-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# --- INICJALIZACJA ---
@st.cache_resource
def init(): return GeometryEngine(), HairstyleRecommender()
engine, recommender = init()

# Stany
if 'stage' not in st.session_state: st.session_state.stage = "IDLE"
if 'data' not in st.session_state: st.session_state.data = None
if 'gender' not in st.session_state: st.session_state.gender = "Female"

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Funkcja rysująca Overlay na wideo
def draw_hud(img, text, progress=0.0, is_active=False):
    h, w, _ = img.shape
    
    # Celownik (Owal)
    cx, cy = w//2, h//2
    # Idealne proporcje do wpasowania twarzy
    ax, ay = int(w/3.8), int(h/2.2)
    
    color = (0, 255, 0) if is_active else (150, 150, 150)
    thickness = 2 if is_active else 1
    
    # Owal
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 200, color, thickness)
    # Krzyż celowniczy
    cv2.line(img, (cx-10, cy), (cx+10, cy), color, 1)
    cv2.line(img, (cx, cy-10), (cx, cy+10), color, 1)

    # Panel informacyjny na górze
    cv2.rectangle(img, (0, 0), (w, 60), (0,0,0), -1)
    cv2.putText(img, text.upper(), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Pasek postępu
    if progress > 0:
        cv2.rectangle(img, (0, 56), (int(w*progress), 60), (0, 255, 0), -1)

# --- LAYOUT APLIKACJI ---

st.markdown('<div class="hero-title">GeoStyle Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Profesjonalna analiza biometryczna twarzy</div>', unsafe_allow_html=True)

# 1. EKRAN STARTOWY
if st.session_state.stage == "IDLE":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>🎥 Rozpocznij sesję</h3>
            <p style="color: #bbb;">Upewnij się, że masz dobre oświetlenie i patrzysz wprost w kamerę.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Wybór płci PRZED skanem (wpływa na algorytm geometrii)
        g = st.selectbox("Wybierz profil analizy:", ["Kobieta", "Mężczyzna"], index=0)
        st.session_state.gender = "Female" if g == "Kobieta" else "Male"
        
        if st.button("URUCHOM SYSTEM", use_container_width=True, type="primary"):
            st.session_state.stage = "FRONT"
            st.rerun()

# 2. EKRAN SKANOWANIA
elif st.session_state.stage in ["FRONT", "LEFT", "RIGHT"]:
    col_main, col_side = st.columns([3, 1])
    
    with col_side:
        st.markdown(f"""
        <div class="glass-card">
            <h4>Instrukcja</h4>
            <p>1. Umieść twarz w owalu.</p>
            <p>2. Wykonaj polecenia wyświetlane na wideo.</p>
            <p>3. Utrzymaj pozycję przez 3 sekundy.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Przerwij", use_container_width=True):
            st.session_state.stage = "IDLE"; st.rerun()

    with col_main:
        # Pętla wideo
        placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            status_text = "Szukam twarzy..."
            progress = 0.0
            is_locked = False
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # Rysowanie siatki (Tesselation) dla efektu Sci-Fi
                mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], 
                                          mp_face_mesh.FACEMESH_TESSELATION, None,
                                          mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))

                # Logika Yaw (Obrót głowy)
                # Obliczamy relację nosa do krawędzi twarzy
                nose_x = landmarks[1].x
                left_x = landmarks[234].x
                right_x = landmarks[454].x
                yaw_ratio = abs(nose_x - left_x) / abs(right_x - left_x)
                
                target = st.session_state.stage
                
                # Warunki pozycji (Tolerancja 10%)
                if target == "FRONT":
                    is_locked = 0.45 < yaw_ratio < 0.55
                    status_text = "Stabilizuj: FRONT" if is_locked else "Ustaw głowę PROSTO"
                elif target == "LEFT":
                    is_locked = yaw_ratio > 0.65
                    status_text = "Stabilizuj: PROFIL LEWY" if is_locked else "Obroc w Prawo"
                elif target == "RIGHT":
                    is_locked = yaw_ratio < 0.35
                    status_text = "Stabilizuj: PROFIL PRAWY" if is_locked else "Obróć w Lewo"

                # Timer (3 sekundy)
                if is_locked:
                    if 'start_hold' not in st.session_state: st.session_state.start_hold = time.time()
                    elapsed = time.time() - st.session_state.start_hold
                    progress = min(elapsed / 2.5, 1.0) # 2.5s dla lepszego UX
                    
                    if elapsed >= 2.5:
                        if target == "FRONT":
                            st.session_state.data = landmarks # Zapisujemy punkty frontowe do analizy
                            st.session_state.stage = "LEFT"
                        elif target == "LEFT":
                            st.session_state.stage = "RIGHT"
                        else:
                            st.session_state.stage = "RESULT"
                        
                        if 'start_hold' in st.session_state: del st.session_state.start_hold
                        cap.release()
                        st.rerun()
                else:
                    if 'start_hold' in st.session_state: del st.session_state.start_hold

            draw_hud(frame, status_text, progress, is_locked)
            placeholder.image(frame, channels="BGR", use_container_width=True)

# 3. EKRAN WYNIKÓW
elif st.session_state.stage == "RESULT":
    # Analiza
    # Poprawka: Przekazujemy płeć do silnika geometrii!
    shape, metrics = engine.get_face_shape(st.session_state.data, gender=st.session_state.gender)
    advice = recommender.get_advice(shape, st.session_state.gender)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <span class="shape-badge">WYNIK</span>
            <div style="margin-top: 10px; color: #888;">Twój kształt to:</div>
            <div class="big-metric">{shape}</div>
            <div style="margin-top: 20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Dopasowanie</span>
                    <span>{int(metrics['match_confidence']*100)}%</span>
                </div>
                <div style="height:8px; background:#333; border-radius:4px;">
                    <div style="height:100%; width:{int(metrics['match_confidence']*100)}%; background:#FF4B4B; border-radius:4px;"></div>
                </div>
            </div>
            <hr style="border-color: #333; margin: 20px 0;">
            <div style="font-size: 0.9rem; color: #aaa;">
                HW Ratio: {metrics['ratio_hw']}<br>
                JF Ratio: {metrics['ratio_jf']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Nowa Analiza", use_container_width=True):
            st.session_state.stage = "IDLE"; st.rerun()

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">💡 Rekomendacje Stylisty</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">{advice['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(f"Top fryzury ({st.session_state.gender})")
        
        # Wyświetlanie kafelków (Flow layout)
        html_tags = "".join([f'<div class="hair-pill">{h}</div>' for h in advice['hairstyles']])
        st.markdown(html_tags, unsafe_allow_html=True)
        
        st.markdown("---")
        st.error(f"⚠️ **Unikaj:** {advice['avoid']}")