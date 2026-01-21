import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap  # 必须在 requirements.txt 中添加 streamlit-shap

# ================= 1. 页面配置与 CSS =================
st.set_page_config(
    page_title="DR MACE Risk Prediction",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stCard { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    /* 建议卡片样式 */
    .advice-card {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.95em;
    }
    .warning-text { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

THRESHOLD = 0.193

# ================= 2. 资源加载 =================
@st.cache_resource
def load_pipeline():
    BASE_DIR = r"D:\A实验室\糖尿病视网膜病变\WEB\WEB文件"
    try:
        with open(os.path.join(BASE_DIR, "Naive_Bayes_Model.pkl"), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "scaler.pkl"), 'rb') as f: scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR, "imputer.pkl"), 'rb') as f: imputer = pickle.load(f)
        with open(os.path.join(BASE_DIR, "feature_meta.pkl"), 'rb') as f: meta = pickle.load(f)
        return model, scaler, imputer, meta
    except Exception as e:
        st.error(f"System Error: Failed to load resources. {e}")
        return None, None, None, None

model, scaler, imputer, meta = load_pipeline()

# ================= 3. 项目介绍 (基于 Manuscript) =================
st.title("🏥 DR Patients MACE Risk Prediction System")
st.markdown("""
**Introduction:**
This clinical decision support system is designed for patients with **Diabetic Retinopathy (DR)** to predict the risk of **Major Adverse Cardiovascular Events (MACE)** within 3 years. 
Developed based on a multi-center cohort of 390 patients, this tool utilizes a **Naive Bayes** machine learning algorithm, which demonstrated superior performance (AUC 0.771) compared to other models. 
It integrates clinical biomarkers (BUN, SBP, HGB), ECG parameters (T-wave abnormalities), and medication history (Statins) to provide individual risk stratification.
""")
st.divider()

# ================= 4. 侧边栏输入 (含 Gender 逻辑) =================
if model:
    with st.sidebar:
        st.header("📋 Patient Parameters")
        
        with st.form("input_form"):
            # --- 新增: 性别输入 (用于动态调整阈值) ---
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
            
            # 3. HGB (动态提示)
            meta_hgb = meta.get('HGB(g/L)', {'min':50, 'max':200, 'mean':125})
            # 根据性别动态显示参考范围
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
    # --- A. 预处理 ---
    try:
        df_input = pd.DataFrame([inputs])
        cols = list(meta.keys()) 
        df_input = df_input[cols]
        
        df_imp = pd.DataFrame(imputer.transform(df_input), columns=cols)
        df_scl = pd.DataFrame(scaler.transform(df_imp), columns=cols)
        
        # 预测概率
        prob = model.predict_proba(df_scl)[:, 1][0]
        
        # 阈值判断 (0.193)
        THRESHOLD = 0.193
        risk_status = "High Risk" if prob >= THRESHOLD else "Low Risk"
        risk_color = "#dc3545" if prob >= THRESHOLD else "#28a745" # Red vs Green
        
    except Exception as e:
        st.error(f"Computation Error: {e}")
        st.stop()

    # --- B. 结果可视化 (Plotly 仪表盘) ---
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📊 MACE Risk Assessment")
        
        # Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"3-Year MACE Probability<br><span style='font-size:0.8em;color:gray'>Threshold: {THRESHOLD*100}%</span>"},
            delta = {'reference': THRESHOLD * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, THRESHOLD * 100], 'color': '#e8f5e9'},  # Green Zone
                    {'range': [THRESHOLD * 100, 100], 'color': '#ffebee'} # Red Zone
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- C. 临床建议 (含性别逻辑) ---
    with col2:
        st.subheader("🩺 Clinical Recommendations")
        
        alerts = []
        # 1. 性别特异性贫血判断
        hgb_thresh = 130 if gender == "Male" else 120
        if inputs['HGB(g/L)'] < hgb_thresh:
            alerts.append(f"🟠 <b>Hemoglobin ({inputs['HGB(g/L)']} g/L):</b> Below normal for {gender} (<{hgb_thresh}). Evaluate for anemia.")
        
        # 2. BUN
        if inputs['BUN(mmol/L)'] > 7.1:
            alerts.append(f"🟡 <b>BUN ({inputs['BUN(mmol/L)']} mmol/L):</b> Elevated. Monitor renal function.")
        
        # 3. T波
        if inputs[t_col] == 1:
            alerts.append("🔴 <b>ECG:</b> T-wave abnormalities detected. High risk marker.")
        
        # 4. 高危干预
        if prob >= THRESHOLD:
            alerts.append(f"🚨 <b>High MACE Risk (>{THRESHOLD*100}%):</b> Intensive management required.")
            if inputs['Statins'] == 0:
                alerts.append("💊 <b>Medication:</b> Consider initiating Statin therapy.")

        if not alerts:
            st.success("✅ All monitored parameters are within acceptable ranges.")
        else:
            for alert in alerts:
                st.markdown(f"<div class='advice-card'>{alert}</div>", unsafe_allow_html=True)

    # --- D. SHAP 交互式解释 (streamlit-shap) ---
    st.divider()
    st.subheader("🔍 Individual Risk Factor Analysis (SHAP)")
    
    with st.spinner("Generating explanation..."):
        try:
            # 构造背景数据 (StandardScaler 均值为0)
            background = pd.DataFrame(np.zeros((1, df_scl.shape[1])), columns=cols)
            
            # Kernel Explainer
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(df_scl, nsamples=50)
            
            # 提取正类 SHAP 值
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base = explainer.expected_value

            # 修正特征名
            clean_names = [n.replace("abnormalities", "") for n in cols]
            
            # 使用 streamlit-shap 显示交互式 Force Plot
            # 如果报错，请确保 requirements.txt 已包含 streamlit-shap
            st_shap(shap.force_plot(
                base, 
                sv, 
                df_scl.iloc[0].values, 
                feature_names=clean_names,
                link="logit"
            ))
            st.caption("Interactive Force Plot: Bars pushing to the right (Red) increase risk, while bars pushing to the left (Blue) decrease risk.")
            
        except Exception as e:
            st.warning(f"Visualization Skipped: {e}")

# ================= 6. 页脚 =================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    Model: Naive Bayes | Threshold: {THRESHOLD} | Based on Manuscript v1.10<br>
    Disclaimer: Clinical support tool only.
</div>
""", unsafe_allow_html=True)

