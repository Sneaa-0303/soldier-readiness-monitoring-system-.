import os
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import folium
from streamlit_folium import st_folium

# --- IST Time Setup ---
IST = timezone(timedelta(hours=5, minutes=30))

def get_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('military_portal.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            badge_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            unit TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS health_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badge_id TEXT,
            timestamp TEXT,
            heart_rate INTEGER,
            spo2 INTEGER,
            body_temp REAL,
            sleep_hours REAL,
            readiness_score INTEGER,
            status TEXT,
            FOREIGN KEY(badge_id) REFERENCES users(badge_id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_pw(password, hashed):
    return hash_pw(password) == hashed

def register_user(badge, name, unit, password, role="Soldier"):
    conn = sqlite3.connect('military_portal.db')
    cur = conn.cursor()
    try:
        cur.execute('INSERT INTO users VALUES (?,?,?,?,?)', 
                    (badge.upper(), name, unit, hash_pw(password), role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def login_user(badge, password):
    conn = sqlite3.connect('military_portal.db')
    cur = conn.cursor()
    cur.execute('SELECT full_name, unit, password_hash, role FROM users WHERE badge_id = ?', (badge.upper(),))
    row = cur.fetchone()
    conn.close()
    if row and verify_pw(password, row[2]):
        return {"full_name": row[0], "unit": row[1], "role": row[3]}
    return None

def save_log(badge, hr, spo2, temp, sleep, score, status):
    conn = sqlite3.connect('military_portal.db')
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO health_logs (badge_id, timestamp, heart_rate, spo2, body_temp, sleep_hours, readiness_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (badge, get_now(), hr, spo2, temp, sleep, score, status))
    conn.commit()
    conn.close()

def fetch_logs(badge):
    conn = sqlite3.connect('military_portal.db')
    df = pd.read_sql_query("SELECT timestamp, heart_rate, spo2, body_temp, sleep_hours, readiness_score, status FROM health_logs WHERE badge_id = ? ORDER BY id DESC", conn, params=(badge,))
    conn.close()
    return df

# --- Page Layout & Styles ---
st.set_page_config(
    page_title="Soldier Readiness & Tactical Command Portal",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .result-card-fit {
        background-color: #0E3A1E;
        border: 2px solid #22C55E;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-top: 15px;
    }
    .result-card-moderate {
        background-color: #3A2E0E;
        border: 2px solid #EAB308;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-top: 15px;
    }
    .result-card-unfit {
        background-color: #450A0A;
        border: 2px solid #EF4444;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Loading ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_pipeline():
    try:
        m_path = os.path.join(APP_DIR, 'soldier_fatigue_model.pkl')
        s_path = os.path.join(APP_DIR, 'scaler.pkl')
        
        m = joblib.load(m_path)
        s = joblib.load(s_path)
        return m, s
    except Exception as err:
        st.warning(f"⚠️ Model load warning: {err}. Fallback evaluation engine active.")
        return None, None

model, scaler = load_pipeline()

# Session state setup
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "test_submitted" not in st.session_state:
    st.session_state.test_submitted = False
if "test_results" not in st.session_state:
    st.session_state.test_results = None

# --- Login / Register Page ---
if not st.session_state.authenticated:
    st.title("🛡️ Defense Forces Telemetry & Readiness Portal")
    st.caption("Centralized Health & Operational Readiness Monitoring System")
    
    t1, t2 = st.tabs(["🔒 Personnel Login", "📝 New Soldier Registration"])
    
    with t1:
        st.subheader("Access Your Dashboard")
        badge_in = st.text_input("Enter Service Badge ID", value="").strip().upper()
        pass_in = st.text_input("Access Password", type="password", key="login_pass")
        
        if st.button("Log In"):
            if badge_in and pass_in:
                res = login_user(badge_in, pass_in)
                if res:
                    st.session_state.authenticated = True
                    st.session_state.badge_id = badge_in
                    st.session_state.user_info = res
                    st.success(f"Welcome back, {res['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid Badge ID or Password.")
            else:
                st.warning("Please fill in all fields.")
                
    with t2:
        st.subheader("Register New Service Personnel")
        reg_badge = st.text_input("Assign Badge ID (e.g., SGT-9012, CPL-4021)").strip().upper()
        reg_name = st.text_input("Full Name (e.g., Cpl. Vikram Rathore)")
        reg_unit = st.selectbox("Deployed Unit / Regiment", [
            "14th Infantry Battalion",
            "9th Para Special Forces",
            "4th Armored Division",
            "Northern Command Signals",
            "High Altitude Warfare School (HAWS)"
        ])
        reg_pass = st.text_input("Set Password", type="password", key="reg_pass")
        confirm_p = st.text_input("Confirm Password", type="password")
        
        if st.button("Register Personnel"):
            if reg_badge and reg_name and reg_pass:
                if reg_pass != confirm_p:
                    st.error("Passwords do not match!")
                else:
                    if register_user(reg_badge, reg_name, reg_unit, reg_pass):
                        st.success(f"Personnel {reg_name} ({reg_badge}) registered successfully!")
                    else:
                        st.error(f"Badge ID '{reg_badge}' is already registered.")
            else:
                st.warning("Please complete all registration fields.")
                
    st.stop()

# --- Main App ---
user = st.session_state.user_info
badge_id = st.session_state.badge_id

# Sidebar
st.sidebar.markdown("### 🪖 Active Personnel")
st.sidebar.write(f"**Badge:** `{badge_id}`")
st.sidebar.write(f"**Name:** {user['full_name']}")
st.sidebar.write(f"**Unit:** {user['unit']}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.test_submitted = False
    st.session_state.test_results = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Threshold Settings")
cutoff = st.sidebar.slider("Safety Cutoff Threshold (%)", 50, 90, 65) / 100.0

st.title("Soldier Readiness Evaluation Portal")
st.caption(f"Personnel: {user['full_name']} ({badge_id}) | Unit: {user['unit']}")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🧪 Take Readiness Test", "📍 Tactical Sector Location", "📜 Medical Log History"])

# --- Tab 1: Readiness Test ---
with tab1:
    st.subheader("1. Enter Telemetry & Optical Data")
    
    col_v, col_o = st.columns(2, gap="large")
    
    with col_v:
        st.markdown("#### Biometric Telemetry Inputs")
        hr = st.slider("Heart Rate (BPM)", 50, 150, 105, key="hr_slider")
        temp = st.slider("Body Temperature (°C)", 35.0, 40.0, 38.0, step=0.1, key="temp_slider")
        spo2 = st.slider("Blood Oxygen (SpO2 %)", 80, 100, 93, key="spo2_slider")
        sleep = st.slider("Rest Duration (Last 24h)", 0, 10, 4, key="sleep_slider")
        active = st.slider("Active Duty Duration (hrs)", 1, 18, 10, key="active_slider")

    with col_o:
        st.markdown("#### Facial Inspection & Voice Check")
        cam_on = st.checkbox("Enable Live Camera Feed", value=False)
        cam_img = None
        
        if cam_on:
            cam_img = st.camera_input("Capture Verification Image", key="camera_input")
            
        st.markdown("---")
        st.caption("🎙️ Voice Analysis (Optional Log)")
        audio_file = st.file_uploader("Upload Voice Telemetry (WAV/MP3)", type=["wav", "mp3"], key="audio_input")
        if audio_file is not None:
            st.audio(audio_file)

    st.markdown("---")
    
    if st.button("🚀 SUBMIT READINESS ASSESSMENT TEST", type="primary", use_container_width=True):
        st.session_state.test_results = None
        
        # Calculate derived metrics
        s_deficit = float(active) / (float(sleep) + 0.1)
        s_index = (float(hr) * float(temp)) / float(spo2)
        
        feats = pd.DataFrame([{
            'heart_rate': float(hr),
            'body_temp': float(temp),
            'sleep_hours': float(sleep),
            'active_hours': float(active),
            'spo2': float(spo2),
            'sleep_deficit': float(s_deficit),
            'stress_index': float(s_index)
        }])
        
        p_fatigue = None
        if model is not None and scaler is not None:
            try:
                X_scaled = scaler.transform(feats)
                p_fatigue = float(model.predict_proba(X_scaled)[0][1])
            except Exception:
                p_fatigue = None

        # Fallback logic if model missing
        if p_fatigue is None:
            r_hr = max(0, (hr - 70) / 80)
            r_spo2 = max(0, (98 - spo2) / 20)
            r_temp = max(0, (temp - 37.0) / 3.0)
            r_sleep = max(0, (8 - sleep) / 8)
            p_fatigue = min(1.0, (r_hr + r_spo2 + r_temp + r_sleep) / 3.5)

        # Image analysis
        v_score = 0.20
        if cam_img is not None:
            raw_img = cv2.imdecode(np.frombuffer(cam_img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 75:
                v_score = 0.75

        # Combined risk calculation
        risk = (0.60 * p_fatigue) + (0.40 * v_score)
        score = max(0, min(100, int((1.0 - risk) * 100)))
        bio_pct = int(p_fatigue * 100)

        # Decision thresholds
        if score >= 70 and risk < cutoff:
            status = "FIT FOR DUTY"
            tier = "FIT"
            msg = "Operational parameters normal. Approved for full deployment."
        elif 45 <= score < 70:
            status = "RESTRICTED / MODERATE DUTY"
            tier = "MODERATE"
            msg = "Moderate strain detected. Restricted to desk/light base duties only."
        else:
            status = "UNFIT FOR DUTY"
            tier = "UNFIT"
            msg = "Severe fatigue / biometric stress detected. Mandatory rest enforced."
        
        st.session_state.test_results = {
            "biometric_risk": bio_pct,
            "visual_risk": int(v_score * 100),
            "readiness_score": score,
            "status": status,
            "tier": tier,
            "guidance": msg,
            "hr": hr,
            "spo2": spo2,
            "temp": temp,
            "sleep": sleep,
            "timestamp": get_now()
        }
        st.session_state.test_submitted = True
        st.rerun()

    # Results view
    if st.session_state.test_submitted and st.session_state.test_results is not None:
        res = st.session_state.test_results
        
        st.markdown("### 📋 Generated Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Biometric Strain Risk", f"{res['biometric_risk']}%")
        col2.metric("Visual Drowsiness Index", f"{res['visual_risk']}%")
        col3.metric("Overall Readiness Score", f"{res['readiness_score']}%")

        if res["tier"] == "FIT":
            st.markdown(f"""
                <div class="result-card-fit">
                    <h2>✅ VERDICT: FIT FOR DUTY</h2>
                    <p><strong>Personnel:</strong> {user['full_name']} ({badge_id})</p>
                    <p><strong>Status:</strong> {res['guidance']}</p>
                    <p><strong>Assessed Time (IST):</strong> {res['timestamp']}</p>
                </div>
            """, unsafe_allow_html=True)
        elif res["tier"] == "MODERATE":
            st.markdown(f"""
                <div class="result-card-moderate">
                    <h2>⚠️ VERDICT: RESTRICTED / MODERATE DUTY</h2>
                    <p><strong>Personnel:</strong> {user['full_name']} ({badge_id})</p>
                    <p><strong>Status:</strong> {res['guidance']}</p>
                    <p><strong>Assessed Time (IST):</strong> {res['timestamp']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-unfit">
                    <h2>🚨 VERDICT: UNFIT FOR DUTY</h2>
                    <p><strong>Personnel:</strong> {user['full_name']} ({badge_id})</p>
                    <p><strong>Status:</strong> {res['guidance']}</p>
                    <p><strong>Assessed Time (IST):</strong> {res['timestamp']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        c_save, c_reset = st.columns([2, 1])
        with c_save:
            if st.button("💾 Save Test Result to Database Record"):
                save_log(badge_id, res['hr'], res['spo2'], res['temp'], res['sleep'], res['readiness_score'], res['status'])
                st.success("Test record saved to database with IST timestamp!")
        with c_reset:
            if st.button("🔄 Clear & Retake Test"):
                st.session_state.test_submitted = False
                st.session_state.test_results = None
                st.rerun()

# --- Tab 2: Sector Map ---
with tab2:
    st.subheader("Tactical Sector Location Map")
    
    sectors = {
        "14th Infantry Battalion": [28.6139, 77.2090],
        "9th Para Special Forces": [32.7266, 74.8570],
        "4th Armored Division": [26.9124, 75.7873],
        "Northern Command Signals": [32.9183, 75.1416],
        "High Altitude Warfare School (HAWS)": [34.1526, 77.5771]
    }
    
    c_opts, c_map = st.columns([1, 3])
    
    with c_opts:
        st.markdown("#### Sector Control")
        sector_name = st.selectbox(
            "Select Deployment Sector",
            list(sectors.keys()),
            index=list(sectors.keys()).index(user['unit']) if user['unit'] in sectors else 0
        )
        coords = sectors[sector_name]
        st.write(f"**Sector Coordinates:** `{coords[0]}° N, {coords[1]}° E`")
    
    with c_map:
        status_now = "NOT ASSESSED"
        if st.session_state.test_results:
            status_now = st.session_state.test_results["status"]
            
        m = folium.Map(location=coords, zoom_start=10)
        
        marker_color = "green" if "FIT" in status_now and "RESTRICTED" not in status_now else "orange" if "RESTRICTED" in status_now else "red" if "UNFIT" in status_now else "blue"
        
        folium.Marker(
            coords,
            popup=f"Personnel: {user['full_name']}\nBadge: {badge_id}\nStatus: {status_now}",
            tooltip=f"{user['full_name']} ({badge_id}) - {sector_name}",
            icon=folium.Icon(color=marker_color, icon="user", prefix="fa")
        ).add_to(m)
        
        st_folium(m, width=700, height=420)

# --- Tab 3: History ---
with tab3:
    st.subheader(f"Medical History Log — {user['full_name']}")
    df_logs = fetch_logs(badge_id)
    
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True)
        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Official Log (CSV)",
            data=csv_data,
            file_name=f"{badge_id}_medical_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No recorded assessment history. Take a test in Tab 1, submit it, and click 'Save Test Result to Database Record'.")