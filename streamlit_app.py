import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap

# ================= 1. 全局配置与阈值 =================
st.set_page_config(
    page_title="DR-MACE Clinical Prediction Tool",
    page_icon="🏥",
    layout="wide"
)

# 阈值设定 (基于 Manuscript 最佳截断值)
THRESHOLD = 0.193

# ================= 2. 专业 CSS 样式 (复刻 STEMI 风格) =================
st.markdown("""
<style>
    /* 全局背景 */
    .main { background-color: #f8f9fa; }
    
    /* 协议卡片通用样式 */
    .protocol-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* 不同等级的卡片边框 */
    .critical-card { border-left: 5px solid #dc3545; } /* 红 */
    .warning-card { border-left: 5px solid #ffc107; }  /* 黄 */
    .info-card { border-left: 5px solid #17a2b8; }     /* 蓝 */
    .safe-card { border-left: 5px solid #28a745; }     /* 绿 */
    
    /* 标题样式 */
    h4 { margin-top: 0; font-size: 1.1em; font-weight: 600; }
    
    /* 列表样式 */
    ul { padding-left: 20px; margin-bottom: 0; color: #444; font-size: 0.95em; }
    li { margin-bottom: 5px; }
    
    /* 结果大卡片 */
    .result-box {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2.5em; font-weight: bold; }
    .metric-label { color: #666; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ================= 3. 资源加载 =================
@st.cache_resource
def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        # 加载核心组件
        with open(os.path.join(BASE_DIR, "Naive_Bayes_Model.pkl"), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "scaler.pkl"), 'rb') as f: scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR, "imputer.pkl"), 'rb') as f: imputer = pickle.load(f)
        return model, scaler, imputer
    except Exception as e:
        st.error(f"System Initialization Error: {e}")
        return None, None, None

model, scaler, imputer = load_pipeline()

# ================= 4. 项目介绍 (Refined Introduction) =================
st.title("🏥 DR-MACE Risk Stratification System")
st.markdown("### 3-Year Major Adverse Cardiovascular Events Prediction in Diabetic Retinopathy")

# 使用 STEMI 代码的卡片布局来介绍项目
intro_cols = st.columns([2, 3])

with intro_cols[0]:
    st.markdown("""
    <div class='protocol-card info-card'>
        <h4 style='color:#17a2b8;'>Model Specifications</h4>
        <ul>
            <li><b>Algorithm:</b> Naive Bayes Classifier</li>
            <li><b>Cohort:</b> Multi-center DR Registry (N=390)</li>
            <li><b>Performance:</b> AUC 0.771 (Validated)</li>
            <li><b>Outcome:</b> 3-Year MACE (MI, Stroke, CV Death)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with intro_cols[1]:
    st.markdown("""
    <div class='protocol-card warning-card'>
        <h4 style='color:#ffc107;'>Key Predictors & Clinical Logic</h4>
        <ul>
            <li><b>Renal Function:</b> BUN (Blood Urea Nitrogen)</li>
            <li><b>Hemodynamics:</b> SBP (Systolic Blood Pressure)</li>
            <li><b>Hematology:</b> HGB (Hemoglobin) with gender-specific norms</li>
            <li><b>ECG Changes:</b> T-wave abnormalities (Ischemia marker)</li>
            <li><b>Medication:</b> Statin therapy status</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================= 5. 侧边栏：参数录入 =================
if model:
    with st.sidebar:
        st.header("📋 Patient Demographics & Labs")
        
        with st.form("input_form"):
            # 性别 (决定 HGB 阈值)
            st.markdown("**Demographics**")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            
            st.markdown("---")
            st.markdown("**Laboratory & Vitals**")
            
            # BUN
            inputs = {}
            inputs['BUN(mmol/L)'] = st.number_input(
                "Blood Urea Nitrogen (BUN)", 
                min_value=0.0, max_value=100.0, 
                value=7.0, step=0.1, format="%.2f",
                help="Reference: 2.8-7.1 mmol/L"
            )
            
            # SBP
            inputs['SBP(mmHg)'] = st.number_input(
                "Systolic BP (SBP)",
                min_value=50, max_value=250, 
                value=130, step=1,
                help="Target: <140 mmHg (General), <130 mmHg (Intensive)"
            )
            
            # HGB (动态参考值)
            hgb_ref = "130-175" if gender == "Male" else "120-155"
            inputs['HGB(g/L)'] = st.number_input(
                f"Hemoglobin (Ref: {hgb_ref})",
                min_value=30, max_value=250, 
                value=135 if gender == "Male" else 125, 
                step=1,
                help="Anemia Screening Parameter"
            )
            
            st.markdown("---")
            st.markdown("**ECG & Medication**")
            
            # T Wave
            t_col = 'T wave  abnormalities' 
            inputs[t_col] = st.selectbox(
                "T-Wave Abnormalities",
                options=[0, 1],
                format_func=lambda x: "Present (Pathological)" if x == 1 else "Absent (Normal)"
            )
            
            # Statins
            inputs['Statins'] = st.selectbox(
                "Statin Therapy",
                options=[0, 1],
                format_func=lambda x: "On Therapy" if x == 1 else "Naive / Not Prescribed"
            )
            
            run_pred = st.form_submit_button("Run Risk Assessment")

# ================= 6. 核心逻辑与结果展示 =================
if model and run_pred:
    # --- 预处理 ---
    try:
        df_input = pd.DataFrame([inputs])
        cols = ['BUN(mmol/L)', 'SBP(mmHg)', 'HGB(g/L)', 'T wave  abnormalities', 'Statins']
        df_input = df_input[cols]
        
        df_imp = pd.DataFrame(imputer.transform(df_input), columns=cols)
        df_scl = pd.DataFrame(scaler.transform(df_imp), columns=cols)
        
        prob = model.predict_proba(df_scl)[:, 1][0]
        
    except Exception as e:
        st.error(f"Computation Error: {e}")
        st.stop()

    # --- 布局：左侧仪表盘，右侧临床建议 ---
    res_col1, res_col2 = st.columns([2, 3])
    
    # === 左侧：Plotly 仪表盘 (视觉重心) ===
    with res_col1:
        # 定义颜色：不再非红即绿，引入过渡色
        if prob < THRESHOLD:
            gauge_color = "#28a745" # Green
            risk_label = "Low Risk Group"
        else:
            gauge_color = "#dc3545" # Red
            risk_label = "High Risk Group"
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': f"<b>3-Year MACE Probability</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, THRESHOLD * 100], 'color': '#e8f5e9'},
                    {'range': [THRESHOLD * 100, 100], 'color': '#ffebee'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=350, margin=dict(l=30,r=30,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Risk Threshold: {THRESHOLD:.1%} (Based on Youden Index)")

    # === 右侧：临床建议卡片 (专业逻辑) ===
    with res_col2:
        st.markdown("### Clinical Decision Support")
        
        # 1. 高危警示 (Critical)
        if prob >= THRESHOLD:
            st.markdown(f"""
            <div class='protocol-card critical-card'>
                <h4 style='color:#dc3545;'>⚠️ High Risk Criteria Met</h4>
                <ul>
                    <li>Predicted probability (<b>{prob:.1%}</b>) exceeds the threshold of {THRESHOLD:.1%}.</li>
                    <li>Refer to <b>Cardiology</b> for comprehensive cardiovascular assessment.</li>
                    <li>Consider intensive risk factor modification.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='protocol-card safe-card'>
                <h4 style='color:#28a745;'>✅ Low Risk Profile</h4>
                <ul>
                    <li>Current probability (<b>{prob:.1%}</b>) is below the intervention threshold.</li>
                    <li>Continue standard DR follow-up and routine risk factor management.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # 2. 异常指标分析 (Lab Alerts)
        alerts = []
        # 性别特异性贫血
        hgb_limit = 130 if gender == "Male" else 120
        if inputs['HGB(g/L)'] < hgb_limit:
            alerts.append(f"<b>Anemia:</b> HGB {inputs['HGB(g/L)']} g/L (<{hgb_limit}). Evaluate iron status/renal anemia.")
        
        # 肾功能
        if inputs['BUN(mmol/L)'] > 7.1:
            alerts.append(f"<b>Renal Impairment:</b> BUN {inputs['BUN(mmol/L)']} mmol/L. Check eGFR/Creatinine.")
            
        # 心电图
        if inputs[t_col] == 1:
            alerts.append("<b>Ischemia:</b> T-wave abnormalities detected. Correlate with clinical symptoms.")
            
        # 血压
        if inputs['SBP(mmHg)'] >= 140:
            alerts.append(f"<b>Hypertension:</b> SBP {inputs['SBP(mmHg)']} mmHg. Intensify antihypertensive therapy.")

        if alerts:
            alert_html = "".join([f"<li>{a}</li>" for a in alerts])
            st.markdown(f"""
            <div class='protocol-card warning-card'>
                <h4 style='color:#856404;'>Biomarker Alerts</h4>
                <ul>{alert_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # 3. 药物建议 (Medication)
        if prob >= THRESHOLD and inputs['Statins'] == 0:
            st.markdown("""
            <div class='protocol-card info-card'>
                <h4 style='color:#0c5460;'>Medication Optimization</h4>
                <ul>
                    <li><b>Statin Therapy:</b> Patient is High Risk but not on Statins.</li>
                    <li>Guideline recommendation: Initiate moderate-to-high intensity statin.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --- SHAP 解释 (底部) ---
    st.markdown("---")
    st.subheader("🔍 Individual Factor Contribution (SHAP Analysis)")
    
    with st.spinner("Calculating feature importance..."):
        try:
            # 构造背景
            background = pd.DataFrame(np.zeros((1, df_scl.shape[1])), columns=cols)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(df_scl, nsamples=100)
            
            # 提取数据 (兼容 list/array)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base_val = explainer.expected_value
                
            if isinstance(base_val, np.ndarray): base_val = base_val.item()

            # 优化显示名称
            display_names = [
                "BUN (Renal)", "SBP (Pressure)", "HGB (Anemia)", 
                "T-Wave (ECG)", "Statins (Meds)"
            ]
            
            explanation = shap.Explanation(
                values=sv,
                base_values=base_val,
                data=df_scl.iloc[0].values,
                feature_names=display_names
            )
            
            # 渲染 JS 图表
            st_shap(shap.plots.force(explanation, matplotlib=False))
            st.caption("Visualizing the 'Push and Pull' of risk factors. Red bars increase risk; Blue bars decrease risk.")
            
        except Exception as e:
            st.warning(f"Feature analysis unavailable: {e}")

# ================= 7. 页脚 =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85em;'>
    <b>Scientific Reference:</b> <i>Machine Learning for MACE Prediction in Diabetic Retinopathy (Manuscript v1.10)</i><br>
    Model: Naive Bayes (Calibrated) | Validation Cohort: N=390 | AUC: 0.771<br>
    &copy; 2024 Clinical Decision Support System. For Research Use Only.
</div>
""", unsafe_allow_html=True)
