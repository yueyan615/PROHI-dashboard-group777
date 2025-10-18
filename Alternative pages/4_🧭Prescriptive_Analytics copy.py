# you need "pip install streamlit-shap" if you haven't already installed it with the requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap


st.set_page_config(
    page_title="Prescriptive Analytics | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

# load the dataset
file_name = './assets/ObesityDataSet_cleaned.parquet'
df = pd.read_parquet(file_name)

############################ SIDEBAR
### Logo
img1 = './img/logo.svg'
st.logo(img1, size= "large", icon_image=None)  

### Navigation on the page
st.sidebar.write("<br>", unsafe_allow_html=True)
def scroll_to(element_id: str):
    components.html(
        f"""
        <script>
        const t = window.parent.document.getElementById("{element_id}");
        if (t) t.scrollIntoView({{behavior: "smooth", block: "start"}});
        </script>
        """,
        height=0,
    )

with st.sidebar:
    sec = st.radio("Navigate on the page", ["Prescriptive Analytics","SHAP Summary Plot", "SHAP Force Plot"], index=0)
    mapping = {"Prescriptive Analytics": "Prescriptive Analytics", "SHAP Summary Plot": "SHAP Summary Plot", "SHAP Force Plot": "SHAP Force Plot"}
    scroll_to(mapping[sec])


############################ MAIN BODY
st.markdown('<div id="Prescriptive Analytics"></div>', unsafe_allow_html=True)
""" # Prescriptive Analytics"""

"""
This section provides explanations for the model’s predictions using SHAP. Users can see which factors contribute most to obesity classification of their input data from the previous Predictive Analytics section. 
Fill in your data in the Predictive Analytics section first, then navigate back here to see the explanations.
"""

#Import prediction from previous dashboard
# Load model
loaded_model = None 
prediction = None
user_data = None

if "prediction" in st.session_state and "loaded_model" in st.session_state and "user_data" in st.session_state:
    prediction = st.session_state.prediction
    loaded_model = st.session_state.loaded_model
    user_data = st.session_state.user_data
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(user_data)

    st.markdown('<div id="SHAP Summary Plot"></div>', unsafe_allow_html=True)
    st.write("## SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, user_data, max_display=14,  plot_size=[10,4], class_names= ['Insufficient Weight','Normal Weight','Overweight Level I','Overweight Level II', 'Obesity Type I','Obesity Type II', 'Obesity Type III'] , show=False, plot_type="bar")
    st.pyplot(fig)
    st.write("The SHAP summary plot above shows the average impact of each feature on the model's predictions across all classes. Features are ranked by their importance, with the most influential features at the top. The length of each bar indicates the magnitude of the feature's contribution to the prediction, averaged over all samples.")

    st.markdown('<div id="SHAP Force Plot"></div>', unsafe_allow_html=True)
    st.write("## SHAP Force Plot")
    ## st.pyplot(shap.plots.force(explainer.expected_value[prediction[0]], shap_values[:, :, prediction[0]], user_data.iloc[0, :] ,matplotlib=True))
    st_shap(shap.force_plot(explainer.expected_value[prediction[0]], shap_values[:, :, prediction[0]], user_data.iloc[0, :]), height=200, width=965)
    st.write("This plot provides a detailed explanation of the model's prediction for the given specific input data. It visualizes how each feature contributes to pushing the prediction from the base value (the average model output) to the final predicted value for the input. Features that increase the prediction are shown in red, while those that decrease it are shown in blue. The width of each arrow represents the magnitude of the feature's impact on the prediction")


         
    # What‑if / Counterfactual Analysis
    with st.expander("What‑if / Counterfactual Analysis (试验变量影响)"):
        st.write("修改下面任意特征后点击 Run counterfactual 查看新的预测概率与 SHAP 解释。")
        # 保证 user_data 是单行 DataFrame

# ...existing code...
        # 保证 user_data 是单行 DataFrame
        base_row = user_data.iloc[0].copy()

        # 有序映射（列表顺序必须与模型中编码 0,1,2... 对应）
        CATEGORY_MAPS = {
            "Veggie_consumption_freq": ["Never", "Sometimes", "Frequently", "Always"],
            "Water_consumption": ["Less than a liter", "Between 1 and 2 L", "More than 2 L"],
            "Physical_activity": ["I do not", "1 or 2 days", "2 or 4 days", "4 or 5 days"],
            "Food_between_meals_freq": ["No", "Sometimes", "Frequently", "Always"],
            "Screen_time": ["0–2 hours", "3–5 hours", "More than 5 hours"],
            "Main_meals_count": ["Between 1 and 2", "Three", "More than three"],
            "Monitors_calories": ["No", "Yes"],
            "High_caloric_food": ["No", "Yes"],
            "Smokes": ["No", "Yes"],
            "Transportation_mode": ["Walking", "Bike", "Motorbike", "Public Transportation", "Automobile"],
        }

        # 动态生成控件（使用 selectbox 显示可读标签，后台映射回编码）
        edited = {}
        for col in base_row.index:
            val = base_row[col]

            # 优先使用预定义的映射（显示可读标签 -> 存储编码）
            if col in CATEGORY_MAPS:
                labels = CATEGORY_MAPS[col]
                # 确定默认选项索引：若原值为编码则直接用编码，否则查找标签匹配
                try:
                    if isinstance(val, (int, np.integer)) and 0 <= int(val) < len(labels):
                        default_idx = int(val)
                    else:
                        default_idx = labels.index(str(val)) if str(val) in labels else 0
                except Exception:
                    default_idx = 0
                chosen = st.selectbox(f"{col}", options=labels, index=default_idx, key=f"cf_{col}")
                # 存回编码（模型通常期待数值编码）
                edited[col] = labels.index(chosen)

            # 数值型字段使用 number_input
            elif pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float, np.integer, np.floating)):
                try:
                    v0 = float(val)
                except Exception:
                    v0 = 0.0
                edited[col] = st.number_input(f"{col}", value=v0, format="%.3f", key=f"cf_{col}")

            # 其他未映射的类别依然用 selectbox 显示原始选项（保留原值类型尽量一致）
            else:
                opts = df[col].dropna().unique().astype(str).tolist()
                default_idx = opts.index(str(val)) if str(val) in opts else 0
                chosen = st.selectbox(f"{col}", options=opts, index=default_idx, key=f"cf_{col}")
                # 尝试恢复原始 dtype（若原来是数字编码的字符串）
                if isinstance(val, (int, np.integer)):
                    try:
                        edited[col] = int(chosen)
                    except Exception:
                        edited[col] = chosen
                elif isinstance(val, (float, np.floating)):
                    try:
                        edited[col] = float(chosen)
                    except Exception:
                        edited[col] = chosen
                else:
                    edited[col] = chosen
# ...existing code...



        if st.button("Run counterfactual"):
            try:
                # 构造单行 DataFrame（保持列顺序）
                cf_row = pd.DataFrame([edited], columns=base_row.index)

                # 类型尝试转换回原 dtypes（简单尝试）
                for c in base_row.index:
                    orig_dtype = base_row[c].__class__
                    if pd.api.types.is_numeric_dtype(type(base_row[c])):
                        cf_row[c] = pd.to_numeric(cf_row[c], errors='coerce')
                    else:
                        cf_row[c] = cf_row[c].astype(type(base_row[c]).__name__ if hasattr(base_row[c], '__class__') else str)

                # 计算原始与反事实概率
                def get_proba(model, X):
                    if hasattr(model, "predict_proba"):
                        return model.predict_proba(X)[0] * 100
                    else:
                        # fallback: use decision_function + softmax
                        scores = model.decision_function(X)[0]
                        exp = np.exp(scores - np.max(scores))
                        return exp / exp.sum() * 100

                orig_probs = get_proba(loaded_model, user_data)
                cf_probs = get_proba(loaded_model, cf_row)

                # 类标签顺序（与你模型训练一致）
                class_labels = ['Insufficient Weight','Normal Weight','Overweight Level I',
                                'Overweight Level II','Obesity Type I','Obesity Type II','Obesity Type III']
                PALETTE = ["#0072b2","#e69f00","#009e73","#cc79a7","#f0e442","#d55e00","#56b4e9"]

                # 显示比较条形图（Plotly）
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(x=orig_probs, y=class_labels, orientation='h',
                                     name='Original', marker_color='lightgray', text=[f"{p:.1f}%" for p in orig_probs],
                                     textposition='outside'))
                fig.add_trace(go.Bar(x=cf_probs, y=class_labels, orientation='h',
                                     name='Counterfactual', marker_color=PALETTE,
                                     text=[f"{p:.1f}%" for p in cf_probs], textposition='outside'))
                fig.update_layout(barmode='group', height=420, title="Original vs Counterfactual Probabilities",
                                  xaxis_title="Probability (%)", margin=dict(l=200, r=30, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # 显示变化表
                delta = cf_probs - orig_probs
                delta_df = pd.DataFrame({
                    "class": class_labels,
                    "orig_%": np.round(orig_probs, 2),
                    "cf_%": np.round(cf_probs, 2),
                    "delta_%": np.round(delta, 2)
                }).sort_values("delta_%", ascending=False)
                st.write("Probability changes (sorted):")
                st.dataframe(delta_df.style.format({"orig_%":"{:.2f}","cf_%":"{:.2f}","delta_%":"{:+.2f}"}))

                # 计算并展示反事实的 SHAP force plot（更直观地看贡献变化）
                shap_vals_cf = explainer.shap_values(cf_row)
                pred_cf = np.argmax(cf_probs)  # 预测类索引
                st.write("## SHAP Force Plot (Counterfactual)")
                st_shap(shap.force_plot(explainer.expected_value[pred_cf], shap_vals_cf[:, :, pred_cf], cf_row.iloc[0, :]), height=200, width=965)

            except Exception as e:
                st.error(f"Counterfactual failed: {e}")
# ...existing code...




else:
    st.warning("Fill in your data in the Predictive Analytics section first, then navigate back here to see the explanations.")
