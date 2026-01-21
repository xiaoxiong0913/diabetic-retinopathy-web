import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap

# ================= 1. 页面配置与 CSS 样式 (装修核心) =================
st.set_page_config(
    page_title="DR MACE Prediction System",
    page_icon="👁️",
    layout="wide"
)

# 注入 STEMI 风格的 CSS
st.markdown("""
<style>
    /* 全局字体与卡片风格 */
    .main { background-color: #f5f7f9; }

    .stCard {
        border-radius: 12px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* 风险卡片样式 */
    .risk-high {
        background-color: #fff5f5;
        border-left: 6px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #f0fff4;
        border-left: 6px solid #28a745;
        padding: 20px;
        border-radius: 8px;
    }

    /* 建议卡片样式 */
    .advice-card {
        background-color: #e3f2fd;
        border-left: 6px solid #2196f3;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }

    /* 标题样式 */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .metric-label { font-size: 0.9em; color: #666; }
    .metric-value { font-size: 1.5em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ================= 2. 核心资源加载 (稳健后端) =================
@st.cache_resource
def load_pipeline():
    # 修改为您实际的文件路径
    BASE_DIR = r"D:\A实验室\糖尿病视网膜病变\WEB\WEB文件"

    try:
        with open(os.path.join(BASE_DIR, "Naive_Bayes_Model.pkl"), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR, "imputer.pkl"), 'rb') as f:
            imputer = pickle.load(f)
        with open(os.path.join(BASE_DIR, "feature_meta.pkl"), 'rb') as f:
            meta = pickle.load(f)
        return model, scaler, imputer, meta
    except Exception as e:
        st.error(f"System Error: Failed to load model resources. {e}")
        return None, None, None, None


model, scaler, imputer, meta = load_pipeline()

# ================= 3. 侧边栏：交互式输入 (手动优化版) =================
if model:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=60)
        st.title("Patient Data")
        st.markdown("Please input clinical parameters:")

        with st.form("input_form"):
            inputs = {}

            # 使用 meta 中的范围作为安全边界，但手动美化 Label 和 Step
            # 1. BUN
            meta_bun = meta.get('BUN(mmol/L)', {'min': 0, 'max': 50, 'mean': 5})
            inputs['BUN(mmol/L)'] = st.slider(
                "Blood Urea Nitrogen (BUN)",
                min_value=0.0,
                max_value=float(meta_bun['max']) + 10,  # 稍微放宽上限
                value=float(meta_bun['mean']),
                format="%.2f mmol/L",
                help="Reference: 2.8-7.1 mmol/L"
            )

            # 2. SBP
            meta_sbp = meta.get('SBP(mmHg)', {'min': 80, 'max': 200, 'mean': 120})
            inputs['SBP(mmHg)'] = st.slider(
                "Systolic Blood Pressure (SBP)",
                min_value=50.0,
                max_value=250.0,
                value=float(meta_sbp['mean']),
                step=1.0,
                format="%d mmHg"
            )

            # 3. HGB
            meta_hgb = meta.get('HGB(g/L)', {'min': 50, 'max': 200, 'mean': 130})
            inputs['HGB(g/L)'] = st.slider(
                "Hemoglobin (HGB)",
                min_value=40.0,
                max_value=220.0,
                value=float(meta_hgb['mean']),
                step=1.0,
                format="%d g/L",
                help="Anemia threshold: <130 g/L (Men), <120 g/L (Women)"
            )

            # 4. T Wave (Categorical)
            # 注意：需严格匹配 features.txt 中的列名 (双空格)
            t_col = 'T wave  abnormalities'
            inputs[t_col] = st.selectbox(
                "ECG: T-Wave Abnormalities",
                options=[0, 1],
                format_func=lambda x: "Present (High Risk)" if x == 1 else "Absent (Normal)",
                index=0
            )

            # 5. Statins (Categorical)
            inputs['Statins'] = st.radio(
                "Statin Therapy Status",
                options=[0, 1],
                format_func=lambda x: "On Therapy" if x == 1 else "No Statin",
                horizontal=True
            )

            submit_btn = st.form_submit_button("🚀 Run Risk Assessment")

# ================= 4. 主界面：预测逻辑与展示 =================
if model and submit_btn:
    # --- A. 数据预处理 (Pipeline) ---
    try:
        df_input = pd.DataFrame([inputs])
        # 严格的列顺序对齐
        cols = list(meta.keys())
        df_input = df_input[cols]

        # 1. Impute (防呆)
        df_imp = pd.DataFrame(imputer.transform(df_input), columns=cols)
        # 2. Scale (标准化)
        df_scl = pd.DataFrame(scaler.transform(df_imp), columns=cols)

        # 3. 预测
        prob = model.predict_proba(df_scl)[:, 1][0]
        pred_cls = model.predict(df_scl)[0]

    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.stop()

    # --- B. 结果展示区 (STEMI 风格) ---
    st.markdown("### 📊 Assessment Report")

    # 动态样式
    risk_css = "risk-high" if prob >= 0.147 else "risk-low"  # 假设阈值 14.7%
    risk_title = "HIGH RISK WARNING" if prob >= 0.147 else "LOW RISK"
    risk_color = "#dc3545" if prob >= 0.147 else "#28a745"

    st.markdown(f"""
    <div class='{risk_css}'>
        <h2 style='color: {risk_color}; margin-top:0;'>{risk_title}</h2>
        <p style='font-size: 1.2rem; margin-bottom:5px;'>Predicted MACE Probability (3-Year):</p>
        <div style='font-size: 3rem; font-weight: bold; color: {risk_color};'>
            {prob:.1%}
        </div>
        <p style='color: gray;'>Based on Naive Bayes probabilistic model</p>
    </div>
    """, unsafe_allow_html=True)

    # --- C. 临床建议逻辑 (根据您的 STEMI 代码逻辑定制) ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🩺 Clinical Alerts & Actions")

        alerts = []
        # 1. T波异常
        if inputs[t_col] == 1:
            alerts.append(
                "🔴 <b>ECG Alert:</b> T-wave abnormality detected. Recommend cardiology consult and continuous ECG monitoring.")

        # 2. 贫血检查
        if inputs['HGB(g/L)'] < 120:
            alerts.append("🟠 <b>Hematology:</b> Low Hemoglobin. Consider anemia workup (Iron, B12, Folate).")

        # 3. 肾功能
        if inputs['BUN(mmol/L)'] > 7.1:
            alerts.append("🟡 <b>Renal:</b> Elevated BUN. Check Creatinine/eGFR and hydration status.")

        # 4. 血压
        if inputs['SBP(mmHg)'] > 140:
            alerts.append("🟡 <b>Vascular:</b> Systolic BP elevated. Optimize antihypertensive therapy.")

        # 5. 他汀类药物建议
        if prob >= 0.20 and inputs['Statins'] == 0:
            alerts.append(
                "🟢 <b>Medication:</b> High MACE risk detected. Consider initiating Statin therapy per guidelines.")

        if not alerts:
            st.success("No critical parameter alerts detected.")
        else:
            for alert in alerts:
                st.markdown(f"<div class='advice-card'>{alert}</div>", unsafe_allow_html=True)

    # --- D. 可解释性 (SHAP) ---
    with col2:
        st.markdown("#### 🔍 Risk Factor Analysis (SHAP)")
        with st.spinner("Calculating feature impact..."):
            try:
                # 构造背景数据 (Zero matrix since scaled mean=0)
                background = pd.DataFrame(np.zeros((1, df_scl.shape[1])), columns=cols)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_vals = explainer.shap_values(df_scl, nsamples=50)

                # 获取正类 SHAP 值
                target_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                base_val = explainer.expected_value[1] if isinstance(shap_vals, list) else explainer.expected_value

                # 修正特征名显示 (加换行)
                display_names = [n.replace("abnormalities", "\nabnorm") for n in cols]

                # 绘制瀑布图
                exp = shap.Explanation(
                    values=target_shap,
                    base_values=base_val,
                    data=df_scl.iloc[0].values,
                    feature_names=display_names
                )

                fig, ax = plt.subplots(figsize=(5, 4))
                shap.plots.waterfall(exp, max_display=5, show=False)
                st.pyplot(fig)
                st.caption("Waterfall plot shows how each value pushes the risk higher (Red) or lower (Blue).")

            except Exception as e:
                st.warning("SHAP calculation skipped (Model format mismatch).")

# ================= 5. 页脚 =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    <b>Disclaimer:</b> This tool is for clinical decision support only and does not replace professional medical judgment.<br>
    Model Version: Naive Bayes v1.0 | Data Source: Multi-center Retinopathy Registry
</div>
""", unsafe_allow_html=True)