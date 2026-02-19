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
st.set_page_config(page_title="GeoStyle AI", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# --- STYLIZACJA (CSS) ---
st.markdown("""
<style>
    /* Główne tło */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(10, 10, 10) 90%);
        color: #ffffff;
    }
    
    /* 2. UKRYWANIE ELEMENTÓW SYSTEMOWYCH */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }   
    
    /* Nagłówek (tylko dla IDLE/RESULT) */
    .hero-title {
        font-size: 4rem;
        font-weight: 850;
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
        margin-bottom: 2rem;
    }

    /* Karty */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 1.5rem;
    }
    
    /* 4. PRZYCISK STARTOWY (GIGANTYCZNY) */
    .big-button button {
        width: 100%;
        height: 100px;       /* ZWIĘKSZONO z 65px */
        font-size: 2.2rem;   /* ZWIĘKSZONO z 1.3rem */
        font-weight: 900;    /* Extra bold */
        background: linear-gradient(90deg, #FF4B4B, #FF914D);
        border: none;
        border-radius: 20px; /* Bardziej zaokrąglony */
        color: white;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 10px 40px rgba(255, 75, 75, 0.4); /* Dodano silny cień/poświatę */
        transition: all 0.2s;
    }
    .big-button button:hover {
        transform: scale(1.03);
        box-shadow: 0 15px 50px rgba(255, 75, 75, 0.6);
    }

    /* 5. STICKY HUD (POWIĘKSZONY ZNACZNIE) */
    .sticky-hud {
        position: fixed;
        top: 30px;
        right: 30px;
        width: 380px; /* Zwiększona szerokość */
        background: rgba(15, 15, 20, 0.95);
        border: 1px solid rgba(100, 100, 100, 0.5);
        border-radius: 16px;
        padding: 30px; /* Większy padding */
        z-index: 9998;
        box-shadow: 0 15px 40px rgba(0,0,0,0.7);
    }
            
    /* 6. CZERWONY PRZYCISK PRZERWIJ (POD HUDEM) */
    div:has(div#fix-stop-btn) {
        display: none;
    }
    
    /* Magiczny selektor: Znajdź przycisk, który jest zaraz po naszym znaczniku */
    div:has(div#fix-stop-btn) + div button {
        position: fixed !important;
        top: 340px !important; /* Zaraz pod HUDem (30px top + ~280px hud height + margin) */
        right: 30px !important;
        width: 380px !important; /* Szerokość taka sama jak HUD */
        height: 60px !important;
        z-index: 99999 !important;
        background-color: rgba(220, 38, 38, 0.9) !important;
        border: 2px solid #EF4444 !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        border-radius: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
    }
    
    div:has(div#fix-stop-btn) + div button:hover {
        background-color: #FF0000 !important;
        border-color: #FF4444 !important;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.6) !important;
        transform: scale(1.02) !important;
        color: white !important;
    }
    
    div:has(div#fix-stop-btn) + div button:active {
        transform: scale(0.98) !important;
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
    
    /* Odstęp dla strony wyników */
    .results-spacer {
        height: 8vh;
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
    
    # Celownik (Owal) - Wycentrowany
    cx, cy = w//2, h//2
    ax, ay = int(w/5.0), int(h/3.0) 
    
    color_active = (0, 255, 0)
    color_idle = (120, 120, 120)
    color = color_active if is_active else color_idle
    thickness = 2 if is_active else 1
    
    # Owal i celownik
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, color, thickness)
    cv2.line(img, (cx-10, cy), (cx+10, cy), color, 1)
    cv2.line(img, (cx, cy-10), (cx, cy+10), color, 1)

   # INSTRUKCJA (Overlay na dole wideo)
    overlay_h = 60
    cv2.rectangle(img, (0, 0), (w, overlay_h), (0,0,0), -1)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.8
    font_thick = 1
    
    text = text.upper()
    tsz = cv2.getTextSize(text, font, font_scale, font_thick)[0]
    tx = (w - tsz[0]) // 2
    ty = int(overlay_h / 2) + 10
    
    cv2.putText(img, text, (tx, ty), font, font_scale, (255,255,255), font_thick, cv2.LINE_AA)
    
    if progress > 0:
        cv2.rectangle(img, (0, overlay_h-5), (int(w*progress), overlay_h), (0, 255, 0), -1)

# --- LOGIKA APLIKACJI ---

# 1. EKRAN STARTOWY
if st.session_state.stage == "IDLE":
    st.markdown('<div class="hero-title">GeoStyle AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Profesjonalna analiza biometryczna twarzy</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>🎥 Zaawansowana analiza biometryczna 3D</h3>
            <p style="color: #bbb;">Upewnij się, że masz dobre oświetlenie i patrzysz wprost w kamerę.</p>
        </div>
        """, unsafe_allow_html=True)
        
        g = st.selectbox("Wybierz profil analizy:", ["Kobieta", "Mężczyzna"], index=0)
        st.session_state.gender = "Female" if g == "Kobieta" else "Male"
        
        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        if st.button("ROZPOCZNIJ SKANOWANIE 🚀"):
            st.session_state.stage = "SCANNING"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# 2. EKRAN SKANOWANIA (SCANNING)
elif st.session_state.stage == "SCANNING":
    
    # UKŁAD: Kamera po lewej (Większa - 66%), Panel prawy (Pusty 33%)
    col_Cam, col_Empty = st.columns([2, 1])

    # --- PRZYCISK PRZERWIJ (PŁYWAJĄCY) ---
    st.markdown('<div id="fix-stop-btn"></div>', unsafe_allow_html=True)
    if st.button("PRZERWIJ SKANOWANIE"):
        st.session_state.stage = "IDLE"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


    with col_Cam:
        placeholder = st.empty()
        hud_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        internal_stage = "FRONT"
        scan_data = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # --- STICKY HUD (HTML) ---
            instr_detail = ""
            if internal_stage == "FRONT": instr_detail = "Patrz prosto w kamerę"
            elif internal_stage == "LEFT": instr_detail = "Pokaż prawy profil"
            elif internal_stage == "RIGHT": instr_detail = "Pokaż lewy profil"
            
            # Powiększone czcionki w HUD
            hud_html = f"""
            <div class="sticky-hud">
                <div style="color:#888; font-size:0.9rem; letter-spacing:2px; margin-bottom:10px;">ETAP SKANOWANIA</div>
                <div style="font-size:2.5rem; font-weight:900; color:#FFF; margin-bottom:15px; line-height: 1;">{internal_stage}</div>
                <div style="font-size:1.2rem; line-height:1.5; color:#DDD;">{instr_detail}</div>
                <div style="margin-top:25px; font-size:1rem; color:#FF4B4B; font-weight:800;">
                    ⏳ TRZYMAJ NIERUCHOMO (3s)
                </div>
            </div>
            """
            hud_placeholder.markdown(hud_html, unsafe_allow_html=True)

            status_text = ""
            progress = 0.0
            is_locked = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], 
                                          mp_face_mesh.FACEMESH_FACE_OVAL, None,
                                          mp_drawing.DrawingSpec(color=(0,255,255), thickness=2))

                nose_x = landmarks[1].x
                left_x = landmarks[234].x
                right_x = landmarks[454].x
                yaw = abs(nose_x - left_x) / abs(right_x - left_x)
                
                if internal_stage == "FRONT":
                    is_locked = 0.45 < yaw < 0.55
                    status_text = "STABILIZUJ: FRONT" if is_locked else "USTAW GLOWE PROSTO"
                elif internal_stage == "LEFT":
                    is_locked = yaw > 0.65
                    status_text = "STABILIZUJ: LEWY PROFIL" if is_locked else "OBROC SIE W PRAWO"
                elif internal_stage == "RIGHT":
                    is_locked = yaw < 0.35
                    status_text = "STABILIZUJ: PRAWY PROFIL" if is_locked else "OBROC SIE W LEWO"

                if is_locked:
                    if 'hold' not in st.session_state: st.session_state.hold = time.time()
                    elapsed = time.time() - st.session_state.hold
                    progress = min(elapsed / 3.0, 1.0)
                    
                    if elapsed >= 3.0:
                        if internal_stage == "FRONT":
                            scan_data["front"] = landmarks
                            internal_stage = "LEFT"
                        elif internal_stage == "LEFT":
                            internal_stage = "RIGHT"
                        else:
                            st.session_state.data = scan_data
                            st.session_state.data["front"] = landmarks
                            internal_stage = "DONE"
                            break
                        del st.session_state.hold
                else:
                    if 'hold' in st.session_state: del st.session_state.hold

            draw_hud(frame, status_text, progress, is_locked)
            placeholder.image(frame, channels="BGR", use_container_width=True)
            
        cap.release()
        
        if internal_stage == "DONE":
            st.session_state.stage = "RESULT"
            st.rerun()

# 3. EKRAN WYNIKÓW
elif st.session_state.stage == "RESULT":
    # Odstęp, żeby wynik był na środku ekranu
    st.markdown('<div class="results-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">GeoStyle Pro</div>', unsafe_allow_html=True)
    
    shape, metrics = engine.get_face_shape(st.session_state.data["front"], gender=st.session_state.gender)
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
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Nowa Analiza", use_container_width=True):
            st.session_state.stage = "IDLE"; st.rerun()

    with col2:
        st.subheader(f"Polecane cięcia ({st.session_state.gender})")
        html_tags = "".join([f'<div class="hair-pill">{h}</div>' for h in advice['hairstyles']])
        st.markdown(html_tags, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info(f"**Charakterystyka:** {advice['description']}")
        st.error(f"**Unikaj:** {advice['avoid']}")