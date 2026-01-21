import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap

# ================= 1. 全局配置与变量 =================
st.set_page_config(
    page_title="DR MACE Risk Prediction",
    page_icon="👁️",
    layout="wide"
)

# 【重要修复1】将阈值定义为全局变量，防止页脚报错
THRESHOLD = 0.193

# 注入 CSS 样式
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stCard { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .advice-card {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# ================= 2. 资源加载 (路径修复) =================
@st.cache_resource
def load_pipeline():
    # 【重要修复2】使用相对路径
    # os.path.dirname(__file__) 会自动获取当前脚本所在的目录（即 GitHub 仓库根目录）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # 加载模型 (请确保 GitHub 仓库里文件名大小写完全一致)
        with open(os.path.join(BASE_DIR, "Naive_Bayes_Model.pkl"), 'rb') as f: 
            model = pickle.load(f)
        
        with open(os.path.join(BASE_DIR, "scaler.pkl"), 'rb') as f: 
            scaler = pickle.load(f)
            
        with open(os.path.join(BASE_DIR, "imputer.pkl"), 'rb') as f: 
            imputer = pickle.load(f)
            
        with open(os.path.join(BASE_DIR, "feature_meta.pkl"), 'rb') as f: 
            meta = pickle.load(f)
            
        return model, scaler, imputer, meta
        
    except FileNotFoundError as e:
        st.error(f"System Error: File not found. Please ensure all .pkl files are uploaded to GitHub root directory. Details: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"System Error: Failed to load resources. {e}")
        return None, None, None, None

model, scaler, imputer, meta = load_pipeline()

# ================= 3. 项目介绍 =================
st.title("🏥 DR Patients MACE Risk Prediction System")
st.markdown("""
**Introduction:**
This clinical decision support system is designed for patients with **Diabetic Retinopathy (DR)** to predict the risk of **Major Adverse Cardiovascular Events (MACE)** within 3 years. 
Developed based on a multi-center cohort, this tool utilizes a **Naive Bayes** machine learning algorithm.
""")
st.divider()

# ================= 4. 侧边栏输入 =================
if model:
    with st.sidebar:
        st.header("📋 Patient Parameters")
        
        with st.form("input_form"):
            # 性别输入
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            
            inputs = {}
            
            # 1. BUN
            meta_bun = meta.get('BUN(mmol/L)', {'min':0, 'max':50, 'mean':7})
            inputs['BUN(mmol/L)'] = st.number_input(
                "Blood Urea Nitrogen (BUN)", 
                min_value=0.0, max_value=100.0, 
                value=float(meta_bun['mean']),
                format="%.2f",
                help="Normal range: 2.8-7.1 mmol/L"
            )
            
            # 2. SBP
            meta_sbp = meta.get('SBP(mmHg)', {'min':80, 'max':200, 'mean':130})
            inputs['SBP(mmHg)'] = st.number_input(
                "Systolic Blood Pressure (SBP)",
                min_value=50, max_value=250,
                value=int(meta_sbp['mean']),
                help="Target: <140 mmHg"
            )
            
            # 3. HGB (动态参考值)
            meta_hgb = meta.get('HGB(g/L)', {'min':50, 'max':200, 'mean':125})
            hgb_ref = "130-175 g/L" if gender == "Male" else "120-155 g/L"
            inputs['HGB(g/L)'] = st.number_input(
                f"Hemoglobin ({hgb_ref})",
                min_value=30, max_value=250,
                value=int(meta_hgb['mean']),
                help=f"Anemia threshold: <{hgb_ref.split('-')[0]} g/L"
            )
            
            # 4. T Wave
            t_col = 'T wave  abnormalities' 
            inputs[t_col] = st.selectbox(
                "ECG: T-Wave Abnormalities",
                options=[0, 1],
                format_func=lambda x: "Present (1)" if x == 1 else "Absent (0)"
            )
            
            # 5. Statins
            inputs['Statins'] = st.selectbox(
                "Statin Use",
                options=[0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
            )
            
            submit_btn = st.form_submit_button("🚀 Run Prediction")

# ================= 5. 预测逻辑与结果展示 =================
if model and submit_btn:
    try:
        df_input = pd.DataFrame([inputs])
        cols = list(meta.keys()) 
        df_input = df_input[cols]
        
        # 预处理
        df_imp = pd.DataFrame(imputer.transform(df_input), columns=cols)
        df_scl = pd.DataFrame(scaler.transform(df_imp), columns=cols)
        
        # 预测
        prob = model.predict_proba(df_scl)[:, 1][0]
        
        # 风险判断
        risk_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
        
    except Exception as e:
        st.error(f"Computation Error: {e}")
        st.stop()

    # --- 仪表盘 ---
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📊 MACE Risk Assessment")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"3-Year MACE Probability<br><span style='font-size:0.8em;color:gray'>Threshold: {THRESHOLD*100}%</span>"},
            delta = {'reference': THRESHOLD * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, THRESHOLD * 100], 'color': '#e8f5e9'},
                    {'range': [THRESHOLD * 100, 100], 'color': '#ffebee'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD * 100}
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- 临床建议 ---
    with col2:
        st.subheader("🩺 Clinical Recommendations")
        alerts = []
        
        # 性别特异性判断
        hgb_thresh = 130 if gender == "Male" else 120
        if inputs['HGB(g/L)'] < hgb_thresh:
            alerts.append(f"🟠 <b>Hemoglobin ({inputs['HGB(g/L)']} g/L):</b> Below normal for {gender}. Evaluate for anemia.")
        
        if inputs['BUN(mmol/L)'] > 7.1:
            alerts.append(f"🟡 <b>BUN:</b> Elevated. Monitor renal function.")
        
        if inputs[t_col] == 1:
            alerts.append("🔴 <b>ECG:</b> T-wave abnormalities detected.")
            
        if prob >= THRESHOLD:
            alerts.append(f"🚨 <b>High Risk (>{THRESHOLD*100}%):</b> Intensive management required.")
            if inputs['Statins'] == 0:
                alerts.append("💊 <b>Medication:</b> Consider Statin therapy.")

        if not alerts:
            st.success("✅ All parameters within acceptable ranges.")
        else:
            for alert in alerts:
                st.markdown(f"<div class='advice-card'>{alert}</div>", unsafe_allow_html=True)

    # --- SHAP ---
    st.divider()
    st.subheader("🔍 Individual Risk Factor Analysis (SHAP)")
    with st.spinner("Analyzing..."):
        try:
            background = pd.DataFrame(np.zeros((1, df_scl.shape[1])), columns=cols)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(df_scl, nsamples=50)
            
            # 兼容性处理
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base = explainer.expected_value

            clean_names = [n.replace("abnormalities", "") for n in cols]
            
            st_shap(shap.force_plot(base, sv, df_scl.iloc[0].values, feature_names=clean_names, link="logit"))
            
        except Exception as e:
            st.warning(f"SHAP Error: {e}")

# ================= 6. 页脚 =================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    Model: Naive Bayes | Threshold: {THRESHOLD} | Based on Manuscript v1.10<br>
    Disclaimer: Clinical support tool only.
</div>
""", unsafe_allow_html=True)
