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
from core.logger import FeedbackLogger

# KONFIGURACJA STRONY
st.set_page_config(page_title="GeoStyle Pro", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# Wygląd - CSS
st.markdown("""
<style>
    /* Główne tło */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(10, 10, 10) 90%);
        color: #ffffff;
    }
    
    /* UKRYWANIE ELEMENTÓW SYSTEMOWYCH */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }   
    
    /* Nagłówek (IDLE/RESULT) */
    .hero-title {
        font-size: 4rem;
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

    /* STICKY HUD */
    .sticky-hud {
        position: fixed;
        top: 30px;
        right: 40px;
        width: 400px; /* Szeroki panel */
        background: rgba(15, 15, 20, 0.95);
        border: 1px solid rgba(100, 100, 100, 0.5);
        border-radius: 16px;
        padding: 30px;
        z-index: 9998;
        box-shadow: 0 15px 50px rgba(0,0,0,0.8);
    }
            
    /* PRZYCISK PRZERWIJ*/
    div:has(div#fix-stop-btn) + div button {
        position: fixed !important;
        top: 260px !important; /* Wyliczone: 30px (top) + ~250px (HUD) + 40px (gap) */
        right: 70px !important;
        width: 340px !important; /* Taka sama szerokość jak HUD */
        height: 70px !important;
        z-index: 99999 !important;
        background-color: rgba(220, 20, 20, 0.9) !important;
        border: 2px solid #FF0000 !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        border-radius: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
    }
    
    div:has(div#fix-stop-btn) + div button:hover {
        background-color: #FF0000 !important;
        box-shadow: 0 0 25px rgba(255, 0, 0, 0.8) !important;
        transform: scale(1.02) !important;
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
    
    div.stButton > button {
    margin-top:10px;
    height: 1.2em;
    text-align: center;                
    font-size: 50px;
    letter-spacing:1px;
}
    /* DOLNY PANEL INFORMACYJNY (skanowanie) */
    .bottom-info-box {
        position: fixed;
        bottom: 30px;
        right: 40px;
        width: 380px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px 20px;
        color: #9CA3AF;
        font-size: 0.85rem;
        z-index: 9997;
        line-height: 1.4;
    }
    /* DYMKI NA STRONIE WYNIKÓW */
    .stAlert {
        padding: 1.5rem !important;
        border-radius: 15px !important;
    }
    .stAlert div p {
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
    }

    /* Tytuł na wynikach */
    .results-title {
        text-align: center; 
        color: #FF4B4B; 
        font-size: 4.5rem; 
        font-weight: 900;
        margin-top: -10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# INICJALIZACJA
@st.cache_resource
def init(): return GeometryEngine(), HairstyleRecommender(), FeedbackLogger()
engine, recommender, fb_logger = init()

if 'stage' not in st.session_state: st.session_state.stage = "IDLE"
if 'data' not in st.session_state: st.session_state.data = None
if 'gender' not in st.session_state: st.session_state.gender = "Female"
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = False

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

def draw_hud(img, text, progress=0.0, is_active=False):
    h, w, _ = img.shape
    cx, cy = w//2, h//2
    
    ay = int(h / 2.5) 
    ax = int(ay * 0.75) 
    
    color_active = (0, 255, 0)
    color_idle = (120, 120, 120)
    color = color_active if is_active else color_idle
    thickness = 2 if is_active else 1
    
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, color, thickness)
    cv2.line(img, (cx-10, cy), (cx+10, cy), color, 1)
    cv2.line(img, (cx, cy-10), (cx, cy+10), color, 1)

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

# LOGIKA APLIKACJI 

# 1. EKRAN STARTOWY
if st.session_state.stage == "IDLE":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="hero-title">GeoStyle AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Profesjonalna analiza biometryczna twarzy</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>🎥 Zaawansowana Analiza Biometryczna 3D</h3>
            <p style="color: #bbb;">Upewnij się, że masz dobre oświetlenie i patrzysz wprost w kamerę.</p>
        </div>
        """, unsafe_allow_html=True)
        
        g = st.selectbox("Wybierz profil analizy:", ["Kobieta", "Mężczyzna"], index=0)
        st.session_state.gender = "Female" if g == "Kobieta" else "Male"
        
        if st.button("ROZPOCZNIJ SKANOWANIE 🚀", width='stretch', type="primary"):
            st.session_state.stage = "SCANNING"
            st.session_state.feedback_submitted = False # Reset pętli przy nowym skanowaniu
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# 2. EKRAN SKANOWANIA
elif st.session_state.stage == "SCANNING":
    st.empty()
    col_Cam, col_Empty = st.columns([2, 1])
    
    with col_Empty:
        st.markdown('<div id="fix-stop-btn"></div>', unsafe_allow_html=True)
        if st.button("PRZERWIJ SKANOWANIE", key="stop_btn"):
            st.session_state.stage = "IDLE"
            st.rerun()
        st.markdown("""
        <div class="bottom-info-box">
            <b>💡 Wskazówki:</b><br>
            • Zapewnij dobre oświetlenie twarzy.<br>
            • Zdejmij akcesoria (np. okulary) dla większej precyzji.
        </div>
        """, unsafe_allow_html=True)

    with col_Cam:
        placeholder = st.empty()
        hud_placeholder = st.empty()
        
        video_source_env = os.getenv("VIDEO_SOURCE", os.path.join(os.getcwd(), "data", "raw", "test_data.mp4"))
        
        # Konwersja łańcucha znaków na int, jeśli zmienna wskazuje na kamerę fizyczną (np. "0")
        video_source = int(video_source_env) if video_source_env.isdigit() else video_source_env
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            st.error(f"KRYTYCZNY BŁĄD I/O: Brak strumienia wejściowego. Nie można zlokalizować pliku: {video_source}")
            st.stop()
            
        
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        internal_stage = "FRONT"
        scan_data = {}
        
        frames_processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                if frames_processed == 0:
                    st.error(f"BŁĄD DEKODOWANIA: Zlokalizowano zasób {video_source}, ale odczyt pierwszej klatki jest niemożliwy. Plik może być pusty lub uszkodzony.")
                    st.stop()
                break
            
            frames_processed += 1
            frame = cv2.flip(frame, 1)
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            instr_detail = ""
            if internal_stage == "FRONT": instr_detail = "Patrz prosto w kamerę"
            elif internal_stage == "RIGHT": instr_detail = "Pokaż lewy profil"
            elif internal_stage == "LEFT": instr_detail = "Pokaż prawy profil"
            
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
                elif internal_stage == "RIGHT":
                    is_locked = yaw > 0.65
                    status_text = "STABILIZUJ: LEWY PROFIL" if is_locked else "OBROC SIE W PRAWO"
                elif internal_stage == "LEFT":
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
                            st.session_state.data = {"lms": scan_data["front"], "w": fw, "h": fh}
                            st.session_state.stage = "RESULT"
                            break
                        del st.session_state.hold
                else:
                    if 'hold' in st.session_state: del st.session_state.hold

            draw_hud(frame, status_text, progress, is_locked)
            
            # Zapobieganie zniekształceniom obrazu
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            placeholder.image(buffer.tobytes(), use_container_width=True)
            
            # Wymuszenie stabilizacji - 25 klatek na sekundę
            time.sleep(0.04)
            
        cap.release()
        st.rerun()

# 3. EKRAN WYNIKÓW
elif st.session_state.stage == "RESULT":
    st.markdown('<h1 class="hero-title">GeoStyle AI</h1>', unsafe_allow_html=True)
    
    d = st.session_state.data
    
    shape, metrics = engine.get_face_shape(d["lms"], frame_width=d["w"], frame_height=d["h"], gender=st.session_state.gender)
    advice = recommender.get_advice(shape, st.session_state.gender)

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="color:#888; font-size:0.9rem; margin-bottom:10px;">IDENTYFIKACJA KSZTAŁTU</div>
            <div style="font-size:3.5rem; font-weight:800; color:#FF4B4B;">{shape}</div>
            <div style="margin: 25px 0; height:1px; background:#333;"></div>
            <div style="display:flex; justify-content:space-between; color:#AAA; font-size:0.9rem;">
                <span>DOPASOWANIE</span><span>{int(metrics['match_confidence']*100)}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # FEEDBACK LOOP
        if not st.session_state.feedback_submitted:
            st.markdown("<div style='text-align:center; color:#AAA; font-size:0.9rem; margin-bottom: 10px;'>Czy algorytm poprawnie rozpoznał Twój kształt twarzy?</div>", unsafe_allow_html=True)
            fb_c1, fb_c2 = st.columns(2)
            with fb_c1:
                st.markdown("<style>div[data-testid='column']:nth-of-type(1) div.stButton > button { font-size: 1.2rem; height: auto; margin-top: 0; }</style>", unsafe_allow_html=True)
                if st.button("✅ TAK", width='stretch'):
                    fb_logger.log_result(shape, st.session_state.gender, metrics, True)
                    st.session_state.feedback_submitted = True
                    st.rerun()
            with fb_c2:
                st.markdown("<style>div[data-testid='column']:nth-of-type(2) div.stButton > button { font-size: 1.2rem; height: auto; margin-top: 0; }</style>", unsafe_allow_html=True)
                if st.button("❌ NIE", width='stretch'):
                    fb_logger.log_result(shape, st.session_state.gender, metrics, False)
                    st.session_state.feedback_submitted = True
                    st.rerun()
        else:
            st.success("Dane zapisane do bufora logowania. System dziękuje za kalibrację.")
            
        st.write("")
        if st.button("🔄 NOWA ANALIZA", width='stretch'):
            st.session_state.stage = "IDLE"; st.rerun()
    with col2:
        st.markdown(f"### 🔥 Rekomendacje ({st.session_state.gender})")
        st.info(f"**Charakterystyka:** {advice['description']}")
        st.write("**PROPOZYCJE FRYZUR:**")
        cols = st.columns(2)
        for i, h in enumerate(advice['hairstyles']):
            with cols[i % 2]:
                st.markdown(f"<div style='background:#262730; padding:15px; border-radius:10px; margin-bottom:10px; border:1px solid #333;'><b>{h}</b></div>", unsafe_allow_html=True)
        st.write("")
        st.error(f"⚠️ **Unikaj:** {advice['avoid']}")