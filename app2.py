import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Config & CSS
st.set_page_config(page_title="Waste Wisdom AI", page_icon="♻️", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f1f5f9; }
    /* กล่อง Metric สีขาว */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #475569 !important; font-weight: bold !important; }
    
    /* ปรับแต่งปุ่ม */
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        width: 100%;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Data
@st.cache_data
def load_data():
    return pd.read_csv('sustainable_waste_management_dataset_2024.csv')

df = load_data()
global_avg = df['waste_kg'].mean()

# 3. Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.title("Waste Wisdom")
    selected_area = st.selectbox("📍 เลือกเขตพื้นที่", df['area'].unique())
    st.divider()
    st.metric("📊 ค่าเฉลี่ยรวมทุกพื้นที่", f"{global_avg:,.1f} กก.")
    
    # แจ้งเขตที่ขยะน้อยสุด
    min_area = df.groupby('area')['waste_kg'].mean().idxmin()
    st.caption(f"💡 ทราบหรือไม่? เขต {min_area} มีปริมาณขยะน้อยที่สุดในระบบ")

# 4. Main Analytics Header
area_df = df[df['area'] == selected_area].copy()
area_avg = area_df['waste_kg'].mean()

st.header(f"📊 ระบบวิเคราะห์และทำนายขยะ: {selected_area}")

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric(f"เฉลี่ยเขต {selected_area}", f"{area_avg:,.1f} กก.")
m2.metric("เทียบค่าเฉลี่ยรวม", f"{global_avg:,.1f} กก.", 
          delta=f"{area_avg - global_avg:,.1f} กก.", delta_color="inverse")
m3.metric("อัตราขยะล้นถัง", f"{(area_df['overflow'].mean()*100):.1f}%")

st.divider()

# 5. กราฟเปรียบเทียบ (ซ้าย) และ ทำนายผล (ขวา)
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("📅 ขยะรายวัน: เขตคุณ VS ภาพรวม")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    area_daily = area_df.groupby('day_name')['waste_kg'].mean().reindex(days_order)
    global_daily = df.groupby('day_name')['waste_kg'].mean().reindex(days_order)
    
    chart_data = pd.DataFrame({
        f'เขต {selected_area}': area_daily,
        'ค่าเฉลี่ยทุกเขต': global_daily
    })
    # กราฟแท่งคู่ สีเขียวเข้มและเทาเข้ม
    st.bar_chart(chart_data, color=["#059669", "#64748b"])

with col_right:
    st.subheader("🔮 จำลองและทำนาย")
    with st.container(border=True):
        p_pop = st.number_input("จำนวนประชากร", value=int(area_df['population'].iloc[-1]))
        
        c1, c2 = st.columns(2)
        p_temp = c1.slider("อุณหภูมิ (°C)", 15, 45, 30)
        p_rain = c2.slider("ฝน (mm)", 0, 100, 10)
        
        c3, c4 = st.columns(2)
        p_week = c3.checkbox("เสาร์-อาทิตย์")
        p_hol = c4.checkbox("วันหยุดเทศกาล")
        
        p_camp = st.toggle("แคมเปญแยกขยะ")
        
        if st.button("ประมวลผล AI"):
            # ฝึกโมเดลแบบรวดเร็ว
            features = ['population', 'temp_c', 'rain_mm', 'is_weekend', 'is_holiday', 'recycling_campaign']
            model = LinearRegression().fit(df[features], df['waste_kg'])
            
            res = model.predict([[p_pop, p_temp, p_rain, int(p_week), int(p_hol), int(p_camp)]])[0]
            st.success(f"**คาดการณ์ปริมาณขยะ:**\n### {res:,.2f} กก.")
            if p_camp: st.balloons()

st.divider()

# 6. กราฟความแม่นยำ (Actual vs Predicted) ด้านล่างสุด
st.subheader("📉 ตรวจสอบความแม่นยำของ AI (ข้อมูล 30 วันล่าสุด)")
features = ['population', 'temp_c', 'rain_mm', 'is_weekend', 'is_holiday', 'recycling_campaign']
model = LinearRegression().fit(df[features], df['waste_kg'])

recent_df = area_df.tail(30).copy()
recent_df['predicted'] = model.predict(recent_df[features])

comparison_data = recent_df.set_index('date')[['waste_kg', 'predicted']]
comparison_data.columns = ['ข้อมูลจริง (Actual)', 'AI ทำนาย (Predicted)']

# แสดงกราฟเส้นเปรียบเทียบ
st.line_chart(comparison_data, color=["#059669", "#f59e0b"])

with st.expander("📂 ดูตารางข้อมูลดิบ"):
    st.dataframe(area_df.tail(10), use_container_width=True)