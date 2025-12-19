import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. การตั้งค่าหน้าเว็บและดีไซน์ ---
st.set_page_config(page_title="Waste Predictor AI", page_icon="🍃", layout="wide")

# ปรับแต่ง CSS ให้ดูทันสมัย
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-title { font-size: 36px; font-weight: bold; color: #1e293b; margin-bottom: 20px; }
    .prediction-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    div.stButton > button:first-child {
        background-color: #10b981;
        color: white;
        width: 100%;
        border-radius: 8px;
        border: none;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. การจัดการข้อมูล ---
@st.cache_data
def load_and_model():
    df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')
    features = ['population', 'temp_c', 'rain_mm', 'is_weekend', 'is_holiday', 'recycling_campaign']
    X = df[features]
    y = df['waste_kg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return df, model, X_test, y_test, features

df, model, X_test, y_test, features = load_and_model()
y_pred = model.predict(X_test)

# --- 3. ส่วนหัวเรื่อง ---
st.markdown('<p class="main-title">🍃 ระบบวิเคราะห์และทำนายปริมาณขยะอัจฉริยะ</p>', unsafe_allow_html=True)
st.write("เครื่องมือช่วยวางแผนจัดการขยะในเมือง โดยใช้โมเดล Machine Learning วิเคราะห์จากปัจจัยแวดล้อม")

# --- 4. ส่วน Dashboard (Metrics) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("ค่าเฉลี่ยขยะ (กก./วัน)", f"{df['waste_kg'].mean():,.0f}")
with col2:
    st.metric("ความแม่นยำ (R²)", f"{r2_score(y_test, y_pred):.3f}")
with col3:
    st.metric("ประชากรเฉลี่ย", f"{df['population'].mean():,.0f}")
with col4:
    st.metric("แคมเปญรีไซเคิล", f"{df['recycling_campaign'].sum()} ครั้ง")

st.markdown("---")

# --- 5. การจัดเลย์เอาต์หลัก ---
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.markdown("### 🔮 คำนวณปริมาณขยะใหม่")
    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        pop = st.number_input("👤 จำนวนประชากร", value=20000, step=100)
        
        c1, c2 = st.columns(2)
        with c1:
            temp = st.slider("🌡️ อุณหภูมิ (°C)", 15, 40, 28)
            weekend = st.toggle("🗓️ วันเสาร์-อาทิตย์")
        with c2:
            rain = st.slider("🌧️ ปริมาณฝน (mm)", 0, 100, 10)
            holiday = st.toggle("🎉 วันหยุดนักขัตฤกษ์")
            
        campaign = st.checkbox("📢 มีแคมเปญรีไซเคิล", value=False)
        
        if st.button("เริ่มทำนายผล"):
            input_df = pd.DataFrame([[pop, temp, rain, int(weekend), int(holiday), int(campaign)]], columns=features)
            res = model.predict(input_df)[0]
            st.markdown(f"""
                <div style="text-align:center; padding:15px; background:#f0fdf4; border-radius:10px; margin-top:15px;">
                    <small>ผลลัพธ์การคาดการณ์</small>
                    <h2 style="color:#059669; margin:0;">{res:,.2f} kg</h2>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("### 📊 วิเคราะห์ความแม่นยำ")
    fig, ax = plt.subplots(figsize=(8, 6))
    # ใช้ Scatter plot แบบนุ่มนวล
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#10b981")
    # เส้น Regression line
    sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color="#f43f5e", linestyle="--")
    
    ax.set_title("Actual vs. Predicted Waste", fontsize=14)
    ax.set_xlabel("ค่าจริง (kg)")
    ax.set_ylabel("ค่าทำนาย (kg)")
    st.pyplot(fig)

# --- 6. ส่วนตารางข้อมูล ---
with st.expander("📂 ดูชุดข้อมูลดิบ (Dataset)"):
    st.dataframe(df.head(20), use_container_width=True)
