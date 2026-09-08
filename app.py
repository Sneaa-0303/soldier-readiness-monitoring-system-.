
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone, timedelta
import sqlite3
import hashlib

# Define IST Timezone (UTC + 5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# =============================================================================
# 1. DATABASE INITIALIZATION (SQLite)
# =============================================================================
def init_db():
    conn = sqlite3.connect('military_portal.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            badge_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            unit TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    c.execute('''
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

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hash(password, hashed_text):
    return make_hash(password) == hashed_text

def register_user(badge_id, full_name, unit, password, role="Soldier"):
    conn = sqlite3.connect('military_portal.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users VALUES (?,?,?,?,?)', 
                  (badge_id.upper(), full_name, unit, make_hash(password), role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def authenticate_user(badge_id, password):
    conn = sqlite3.connect('military_portal.db')
    c = conn.cursor()
    c.execute('SELECT full_name, unit, password_hash, role FROM users WHERE badge_id = ?', (badge_id.upper(),))
    data = c.fetchone()
    conn.close()
    if data and check_hash(password, data[2]):
        return {"full_name": data[0], "unit": data[1], "role": data[3]}
    return None

def log_health_data(badge_id, hr, spo2, temp, sleep, score, status):
    conn = sqlite3.connect('military_portal.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO health_logs (badge_id, timestamp, heart_rate, spo2, body_temp, sleep_hours, readiness_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (badge_id, get_ist_now_str(), hr, spo2, temp, sleep, score, status))
    conn.commit()
    conn.close()

def get_user_logs(badge_id):
    conn = sqlite3.connect('military_portal.db')
    df = pd.read_sql_query("SELECT timestamp, heart_rate, spo2, body_temp, sleep_hours, readiness_score, status FROM health_logs WHERE badge_id = ? ORDER BY id DESC", conn, params=(badge_id,))
    conn.close()
    return df
# --- DEBUG CHECK ---
st.write("Is Model Loaded?:", model is not None)
st.write("Is Scaler Loaded?:", scaler is not None)
# =============================================================================
# 2. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Soldier Readiness & Tactical Command Portal",
    page_icon="🛡️",
    layout="wide"
)

# Custom Banner CSS
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

# =============================================================================
# 3. MODEL LOAD
# =============================================================================
@st.cache_resource
def load_ml_pipeline():
    try:
        model = joblib.load('soldier_fatigue_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_ml_pipeline()

# Initialize Session State Variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "test_submitted" not in st.session_state:
    st.session_state.test_submitted = False
if "test_results" not in st.session_state:
    st.session_state.test_results = None

# =============================================================================
# 4. AUTHENTICATION (LOGIN & REGISTRATION)
# =============================================================================
if not st.session_state.authenticated:
    st.title("🛡️ Defense Forces Telemetry & Readiness Portal")
    st.caption("Centralized Health & Operational Readiness Monitoring System")
    
    auth_tab1, auth_tab2 = st.tabs(["🔒 Personnel Login", "📝 New Soldier Registration"])
    
    with auth_tab1:
        st.subheader("Access Your Dashboard")
        badge_input = st.text_input("Enter Service Badge ID", value="").strip().upper()
        password_input = st.text_input("Access Password", type="password", key="login_pass")
        
        if st.button("Log In"):
            if badge_input and password_input:
                user_data = authenticate_user(badge_input, password_input)
                if user_data:
                    st.session_state.authenticated = True
                    st.session_state.badge_id = badge_input
                    st.session_state.user_info = user_data
                    st.success(f"Welcome back, {user_data['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid Badge ID or Password.")
            else:
                st.warning("Please fill in all fields.")
                
    with auth_tab2:
        st.subheader("Register New Service Personnel")
        new_badge = st.text_input("Assign Badge ID (e.g., SGT-9012, CPL-4021)").strip().upper()
        new_name = st.text_input("Full Name (e.g., Cpl. Vikram Rathore)")
        new_unit = st.selectbox("Deployed Unit / Regiment", [
            "14th Infantry Battalion",
            "9th Para Special Forces",
            "4th Armored Division",
            "Northern Command Signals",
            "High Altitude Warfare School (HAWS)"
        ])
        new_pass = st.text_input("Set Password", type="password", key="reg_pass")
        confirm_pass = st.text_input("Confirm Password", type="password")
        
        if st.button("Register Personnel"):
            if new_badge and new_name and new_pass:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match!")
                else:
                    success = register_user(new_badge, new_name, new_unit, new_pass)
                    if success:
                        st.success(f"Personnel {new_name} ({new_badge}) registered successfully! You can now log in.")
                    else:
                        st.error(f"Badge ID '{new_badge}' is already registered in the system.")
            else:
                st.warning("Please complete all registration fields.")
                
    st.stop()

# =============================================================================
# 5. DASHBOARD & TEST FORM
# =============================================================================
user = st.session_state.user_info
badge_id = st.session_state.badge_id

# Sidebar Profile
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
alert_threshold = st.sidebar.slider("Safety Cutoff Threshold (%)", 50, 90, 65) / 100.0

st.title("Soldier Readiness Evaluation Portal")
st.caption(f"Personnel: {user['full_name']} ({badge_id}) | Unit: {user['unit']}")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🧪 Take Readiness Test", "📍 Tactical Sector Location", "📜 Medical Log History"])

# --- TAB 1: SUBMISSION FORM & RESULTS ---
with tab1:
    st.subheader("1. Enter Telemetry & Optical Data")
    
    col_vitals, col_optical = st.columns(2, gap="large")
    
    with col_vitals:
        st.markdown("#### Biometric Telemetry Inputs")
        heart_rate = st.slider("Heart Rate (BPM)", 50, 150, 105, key="hr_slider")
        body_temp = st.slider("Body Temperature (°C)", 35.0, 40.0, 38.0, step=0.1, key="temp_slider")
        spo2 = st.slider("Blood Oxygen (SpO2 %)", 80, 100, 93, key="spo2_slider")
        sleep_hours = st.slider("Rest Duration (Last 24h)", 0, 10, 4, key="sleep_slider")
        active_hours = st.slider("Active Duty Duration (hrs)", 1, 18, 10, key="active_slider")

    with col_optical:
        st.markdown("#### Facial Inspection & Voice Check")
        cam_toggle = st.checkbox("Enable Live Camera Feed", value=False)
        captured_img = None
        
        if cam_toggle:
            captured_img = st.camera_input("Capture Personnel Verification Image", key="camera_input")
            
        st.markdown("---")
        st.caption("🎙️ Voice Analysis (Optional Check)")
        audio_file = st.file_uploader("Upload Voice Telemetry Log (WAV/MP3)", type=["wav", "mp3"], key="audio_input")
        if audio_file is not None:
            st.audio(audio_file)

    st.markdown("---")
    
    # --- SUBMIT BUTTON ---
    if st.button("🚀 SUBMIT READINESS ASSESSMENT TEST", type="primary", use_container_width=True):
        # Clear prior results to force re-evaluation
        st.session_state.test_results = None
        
        # Calculate Features
        sleep_deficit = active_hours / (sleep_hours + 0.1)
        stress_index = (heart_rate * body_temp) / spo2
        
        # Calculate Physical Fatigue Probability using ML Model
        physical_fatigue_prob = 0.30
        if model is not None and scaler is not None:
            raw_inputs = np.array([[heart_rate, body_temp, sleep_hours, active_hours, spo2, sleep_deficit, stress_index]])
            scaled_inputs = scaler.transform(raw_inputs)
            physical_fatigue_prob = model.predict_proba(scaled_inputs)[0][1]

        # Calculate Visual Drowsiness Score
        visual_score = 0.20
        if captured_img is not None:
            img = cv2.imdecode(np.frombuffer(captured_img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if brightness < 75:
                visual_score = 0.75

        # Multimodal Score Fusion
        total_risk = (0.60 * physical_fatigue_prob) + (0.40 * visual_score)
        readiness_score = max(0, int((1 - total_risk) * 100))
        is_fit = total_risk < alert_threshold
        
        # Store freshly computed results in Session State with IST timestamp
        st.session_state.test_results = {
            "biometric_risk": int(physical_fatigue_prob * 100),
            "visual_risk": int(visual_score * 100),
            "readiness_score": readiness_score,
            "status": "FIT FOR DUTY" if is_fit else "UNFIT FOR DUTY",
            "is_fit": is_fit,
            "hr": heart_rate,
            "spo2": spo2,
            "temp": body_temp,
            "sleep": sleep_hours,
            "timestamp": get_ist_now_str()
        }
        st.session_state.test_submitted = True
        st.rerun()

    # --- DISPLAY RESULTS CARD AFTER SUBMISSION ---
    if st.session_state.test_submitted and st.session_state.test_results is not None:
        res = st.session_state.test_results
        
        st.markdown("### 📋 Generated Assessment Results")
        
        # Metric Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Biometric Strain Risk", f"{res['biometric_risk']}%")
        m2.metric("Visual Drowsiness Index", f"{res['visual_risk']}%")
        m3.metric("Overall Readiness Score", f"{res['readiness_score']}%")

        if res["is_fit"]:
            st.markdown(f"""
                <div class="result-card-fit">
                    <h2>✅ VERDICT: FIT FOR DUTY</h2>
                    <p><strong>Personnel:</strong> {user['full_name']} ({badge_id})</p>
                    <p><strong>Status:</strong> Operational parameters are within normal safety limits. Ready for active duty deployment.</p>
                    <p><strong>Assessed Time (IST):</strong> {res['timestamp']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-unfit">
                    <h2>🚨 VERDICT: UNFIT FOR DUTY</h2>
                    <p><strong>Personnel:</strong> {user['full_name']} ({badge_id})</p>
                    <p><strong>Status:</strong> High fatigue or biometric stress detected. Mandatory tactical rest enforced immediately.</p>
                    <p><strong>Assessed Time (IST):</strong> {res['timestamp']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_save, col_reset = st.columns([2, 1])
        with col_save:
            if st.button("💾 Save Test Result to Database Record"):
                log_health_data(badge_id, res['hr'], res['spo2'], res['temp'], res['sleep'], res['readiness_score'], res['status'])
                st.success("Test record successfully saved to database with IST timestamp!")
        with col_reset:
            if st.button("🔄 Clear & Retake Test"):
                st.session_state.test_submitted = False
                st.session_state.test_results = None
                st.rerun()

# --- TAB 2: MAP ---
with tab2:
    st.subheader("Tactical Sector Location Map")
    st.caption(f"Sector Location assigned to unit: {user['unit']}")
    
    # Preset Tactical Sectors across India / HQ
    sector_coordinates = {
        "14th Infantry Battalion": [28.6139, 77.2090],          # New Delhi HQ
        "9th Para Special Forces": [32.7266, 74.8570],           # Jammu Sector
        "4th Armored Division": [26.9124, 75.7873],              # Western Desert Sector
        "Northern Command Signals": [32.9183, 75.1416],          # Udhampur Command
        "High Altitude Warfare School (HAWS)": [34.1526, 77.5771] # Leh High Altitude
    }
    
    unit_loc = sector_coordinates.get(user['unit'], [28.6139, 77.2090])
    
    col_map_opts, col_map_disp = st.columns([1, 3])
    
    with col_map_opts:
        st.markdown("#### Sector Map Options")
        selected_sector = st.selectbox(
            "Select Deployed Sector Zone",
            list(sector_coordinates.keys()),
            index=list(sector_coordinates.keys()).index(user['unit']) if user['unit'] in sector_coordinates else 0
        )
        unit_loc = sector_coordinates[selected_sector]
        st.write(f"**Coordinates:** `{unit_loc[0]}° N, {unit_loc[1]}° E`")
    
    with col_map_disp:
        current_status = "NOT ASSESSED"
        if st.session_state.test_results:
            current_status = st.session_state.test_results["status"]
            
        field_map = folium.Map(location=unit_loc, zoom_start=10)
        folium.Marker(
            unit_loc,
            popup=f"Personnel: {user['full_name']}\nBadge: {badge_id}\nStatus: {current_status}",
            tooltip=f"{user['full_name']} ({badge_id}) - {selected_sector}",
            icon=folium.Icon(color="green" if current_status == "FIT FOR DUTY" else "red" if current_status == "UNFIT FOR DUTY" else "blue", icon="user", prefix="fa")
        ).add_to(field_map)
        
        st_folium(field_map, width=700, height=420)

# --- TAB 3: MEDICAL LOG HISTORY ---
with tab3:
    st.subheader(f"Medical History Log — {user['full_name']}")
    history_df = get_user_logs(badge_id)
    
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
        csv_bytes = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Official Log (CSV)",
            data=csv_bytes,
            file_name=f"{badge_id}_medical_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No recorded assessment history. Take a test in Tab 1, submit it, and click 'Save Test Result to Database Record'.")