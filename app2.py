import streamlit as st
import pandas as pd

# 1. Config & CSS (เน้นความสวยงาม กล่องขาว ตัวหนังสือเข้ม)
st.set_page_config(page_title="Waste Wisdom AI", page_icon="♻️", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f1f5f9; }
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #475569 !important; font-weight: bold !important; }
    .stButton>button {
        background-color: #10b981; color: white; border-radius: 12px;
        font-weight: bold; width: 100%; height: 3em;
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

# 4. Main Analytics Header
area_df = df[df['area'] == selected_area].copy()
area_avg = area_df['waste_kg'].mean()

st.header(f"📊 วิเคราะห์และทำนายขยะ: {selected_area}")

m1, m2, m3 = st.columns(3)
m1.metric(f"เฉลี่ยเขต {selected_area}", f"{area_avg:,.1f} กก.")
m2.metric("เทียบค่าเฉลี่ยรวม", f"{global_avg:,.1f} กก.", 
          delta=f"{area_avg - global_avg:,.1f} กก.", delta_color="inverse")
m3.metric("อัตราขยะล้นถัง", f"{(area_df['overflow'].mean()*100):.1f}%")

st.divider()

# 5. กราฟแท่งคู่ (ซ้าย) และ ส่วนทำนาย (ขวา)
col_chart, col_pred = st.columns([1.5, 1])

with col_chart:
    st.subheader("📅 เปรียบเทียบรายวัน (เขตคุณ VS ภาพรวม)")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    area_daily = area_df.groupby('day_name')['waste_kg'].mean().reindex(days_order)
    global_daily = df.groupby('day_name')['waste_kg'].mean().reindex(days_order)
    
    chart_data = pd.DataFrame({
        f'เขต {selected_area}': area_daily,
        'ค่าเฉลี่ยรวมทุกเขต': global_daily
    })
    st.bar_chart(chart_data, color=["#059669", "#64748b"])

with col_pred:
    st.subheader("🔮 จำลองทำนายผล (สูตรสถิติ)")
    with st.container(border=True):
        p_pop = st.number_input("จำนวนประชากร", value=int(area_df['population'].iloc[-1]))
        c1, c2 = st.columns(2)
        p_temp = c1.slider("อุณหภูมิ", 15, 45, 30)
        p_rain = c2.slider("ปริมาณฝน", 0, 100, 10)
        p_camp = st.toggle("เปิดแคมเปญแยกขยะ")
        
        if st.button("ประมวลผลด่วน"):
            # ใช้การคำนวณทางสถิติพื้นฐาน (Base Rate + Population Factor)
            base_rate = area_avg / area_df['population'].mean()
            prediction = p_pop * base_rate
            
            # ปรับตามเงื่อนไข (Heuristics)
            if p_rain > 50: prediction *= 1.1 # ฝนตกขยะเพิ่ม 10%
            if p_temp > 35: prediction *= 1.05 # ร้อนจัดขยะเพิ่ม 5%
            if p_camp: prediction *= 0.85 # แคมเปญลดขยะ 15%
            
            st.success(f"**คาดการณ์:** {prediction:,.2f} กก.")
            if p_camp: st.balloons()

st.divider()

# 6. กราฟแนวโน้มขยะ 30 วันล่าสุด (แทนกราฟความแม่นยำ)
st.subheader("📉 แนวโน้มปริมาณขยะ 30 วันล่าสุด")
recent_df = area_df.tail(30).copy()
st.line_chart(recent_df.set_index('date')['waste_kg'], color="#059669")

with st.expander("📂 ดูตารางข้อมูล"):
    st.dataframe(area_df.tail(10), use_container_width=True)