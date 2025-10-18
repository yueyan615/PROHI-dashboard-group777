import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from streamlit_shap import st_shap

st.set_page_config(
    page_title="Prescriptive Analytics | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.logo('./img/logo.svg', size="large", icon_image=None)

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
    # Navigation radio buttons
    sec = st.radio(
        "Navigate on the page",
        ["SHAP Summary Plot", "SHAP Force Plot", "What-if Analysis", "Top-3 Suggestions"],
        index=0
    )
    mapping = {
        "SHAP Summary Plot": "SHAP Summary Plot",
        "SHAP Force Plot": "SHAP Force Plot",
        "What-if Analysis": "What-if Analysis",
        "Top-3 Suggestions": "Top-3 Suggestions",
    }
    scroll_to(mapping[sec])

# -------------------------------
# Main body
# -------------------------------
st.markdown('<div id="Prescriptive Analytics"></div>', unsafe_allow_html=True)
st.title("Prescriptive Analytics")

st.markdown(
    """
This page explains the latest prediction using **SHAP** and provides a **What-if / Counterfactual** tool.  
First run a prediction in the **Predictive Analytics** page.
"""
)

# -------------------------------
# Read from session state (DO NOT modify Predictive page)
# -------------------------------
loaded_model = st.session_state.get("loaded_model", None)
prediction   = st.session_state.get("prediction", None)      # shape (1,)
user_data    = st.session_state.get("user_data", None)       # encoded row: shape (1, n_features)

if loaded_model is None or prediction is None or user_data is None:
    st.warning("Please complete a prediction in the **Predictive Analytics** page first.")
    st.stop()

# -------------------------------
# Compute SHAP values (robust to binary/multiclass)
# -------------------------------
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(user_data)

def get_per_class_shap(shap_values, class_idx=None):
    """Return (sv, expected) for the selected output."""
    if isinstance(shap_values, (list, tuple)):
        if class_idx is None:
            class_idx = int(prediction[0])
        sv = shap_values[class_idx]                      # (n_samples, n_features)
        expected = explainer.expected_value[class_idx]
    else:
        sv = shap_values                                 # (n_samples, n_features)
        expected = explainer.expected_value
    return sv, expected

def proba_row(model, Xrow: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(Xrow)[0]

# === Scores ===
def severity_score(p: np.ndarray) -> float:
    """Expected class index (0..6). Lower is better."""
    idx = np.arange(len(p))
    return float((p * idx).sum())

def health_score_from_severity(sev: float) -> float:
    """Map severity (0..6) -> Health Score (0..100, higher is better)."""
    return 100.0 * (1.0 - sev / 6.0)

def health_score(p: np.ndarray) -> float:
    """Health Score from probability vector directly."""
    return health_score_from_severity(severity_score(p))

# -------------------------------
# SHAP Summary Plot (same content; smaller fonts/figure)
# -------------------------------

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

else:
    st.warning("Fill in your data in the Predictive Analytics section first, then navigate back here to see the explanations.")

   







# -------------------------------
# What-if / Counterfactual Analysis
# -------------------------------
st.markdown('<div id="What-if Analysis"></div>', unsafe_allow_html=True)
st.markdown("---")
st.subheader("What-if / Counterfactual Analysis")

st.caption(
    "Adjust feasible, encoded factors below to see how the predicted class distribution and your **Health Score (0–100 ↑)** change. "
    "Educational use only—this does **not** imply causality or medical advice."
)

CLASSES = [
    "Insufficient Weight", "Normal Weight", "Overweight Level I",
    "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III"
]

# Actionable features and their allowed values (encoded)
ACTIONABLE_BOUNDS = {
    "Veggie_consumption_freq": [0, 1, 2, 3],
    "Water_consumption": [0, 1, 2],
    "Physical_activity": [0, 2, 3, 4],
    "Food_between_meals_freq": [0, 1, 2, 3],
    "Screen_time": [0, 1, 2],
    "Main_meals_count": [0, 1, 2],
    "Monitors_calories": [0, 1],
    "High_caloric_food": [0, 1],
    "Smokes": [0, 1],
    "Transportation_mode": [1, 2, 3, 4, 5],
    # "Alcohol_consumption_freq": [0, 1, 2, 3],
}

LABELS = {
    "Veggie_consumption_freq": {0: "Never", 1: "Sometimes", 2: "Frequently", 3: "Always"},
    "Water_consumption": {0: "Less than a liter", 1: "Between 1 and 2 L", 2: "More than 2 L"},
    "Physical_activity": {0: "I do not", 2: "1 or 2 days", 3: "2 or 4 days", 4: "4 or 5 days"},
    "Food_between_meals_freq": {0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"},
    "Screen_time": {0: "0–2 hours", 1: "3–5 hours", 2: "More than 5 hours"},
    "Main_meals_count": {0: "Between 1 and 2", 1: "Three", 2: "More than three"},
    "Monitors_calories": {0: "No", 1: "Yes"},
    "High_caloric_food": {0: "No", 1: "Yes"},
    "Smokes": {0: "No", 1: "Yes"},
    "Transportation_mode": {1: "Walking", 2: "Bike", 3: "Motorbike", 4: "Public Transportation", 5: "Automobile"},
}

# Baseline
base_df   = user_data.copy()
base_prob = proba_row(loaded_model, base_df)
base_sev  = severity_score(base_prob)
base_hs   = health_score_from_severity(base_sev)
base_cls  = CLASSES[int(np.argmax(base_prob))]

m1, m2= st.columns(2)
with m1:
    st.metric("Predicted class", base_cls)
with m2:
    st.metric("Health Score (0–100 ↑)", f"{base_hs:.1f}")



# 4-column layout for selectors
st.markdown("### Adjust factors manually")
st.caption("Values shown are **encoded** levels used by the model.")
cf_row = base_df.iloc[0].copy()

def chunk4(items):
    for i in range(0, len(items), 4):
        yield items[i:i+4]

items = list(ACTIONABLE_BOUNDS.items())
for grp in chunk4(items):
    cols = st.columns(len(grp))
    for c, (feat, allowed) in zip(cols, grp):
        if feat not in cf_row.index:
            continue
        with c:
            cur = int(cf_row[feat])
            try:
                idx = allowed.index(cur)
            except ValueError:
                idx = 0

            # >>> make a selectbox with labels if available
            cf_row[feat] = st.selectbox(
                label=feat,
                options=allowed,
                index=idx,
                format_func=lambda v: f"{v} — {LABELS.get(feat, {}).get(v, 'N/A')}",
                key=f"cf_{feat}"
            )

# Recompute with adjustments
cf_df   = pd.DataFrame([cf_row])
cf_prob = proba_row(loaded_model, cf_df)
cf_sev  = severity_score(cf_prob)
cf_hs   = health_score_from_severity(cf_sev)

delta_hs = cf_hs - base_hs   # points (0..100)

left, right = st.columns(2)
with left:
    st.write("**Adjusted class probabilities**")
    st.bar_chart(pd.Series(cf_prob, index=CLASSES))
with right:
    st.write("**Original vs Adjusted Health Score**")
    fig_cmp, ax_cmp = plt.subplots(figsize=(4, 2))
    bars = ax_cmp.barh(["Original", "Adjusted"], [base_hs, cf_hs])
    ax_cmp.set_xlabel("Health Score (higher is better)")
    ax_cmp.bar_label(bars, fmt="%.1f", padding=3)
    plt.tight_layout()
    st.pyplot(fig_cmp, clear_figure=True)
    st.metric("Δ Health Score (Adjusted − Original)", f"{delta_hs:+.1f} pts")

# -------------------------------
# Top-3 Suggestions（Health Score based）
# -------------------------------
st.markdown('<div id="Top-3 Suggestions"></div>', unsafe_allow_html=True)
st.markdown("### Top-3 Single-Step Suggestions (by Health Score ↑)")

st.info(
    r"""
We try **one change at a time** on actionable features and pick the **top-3** with the **largest Health Score increase**.  
Bigger positive Δ means better (↑ Health Score).
"""
)

try:
    # based on health score
    current = base_df.iloc[0].copy()
    suggestions = []

    for feat, allowed in ACTIONABLE_BOUNDS.items():
        if feat not in current.index:
            continue
        cur_val = int(current[feat])
        for alt in allowed:
            alt = int(alt)
            if alt == cur_val:
                continue
            test = current.copy()
            test[feat] = alt
            new_prob = proba_row(loaded_model, pd.DataFrame([test]))
            new_hs   = health_score(new_prob)  # 0-100 越高越好
            delta_hs = float(new_hs - base_hs)
            suggestions.append((feat, cur_val, alt, delta_hs))

    if not suggestions:
        st.info("No actionable single-step suggestions available.")
    else:
        res = (
            pd.DataFrame(
                suggestions,
                columns=["Feature", "Current", "Alternative", "Δ Health Score (pts)"]
            )
            .sort_values("Δ Health Score (pts)", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )

        # make codes more readable
        def fmt_code_with_label(feat: str, val) -> str:
            try:
                ival = int(val)
            except Exception:
                return str(val)
            label = LABELS.get(feat, {}).get(ival)
            return f"{ival} — {label}" if label is not None else f"{ival}"

        res["Current"] = res.apply(lambda r: fmt_code_with_label(r["Feature"], r["Current"]), axis=1)
        res["Alternative"] = res.apply(lambda r: fmt_code_with_label(r["Feature"], r["Alternative"]), axis=1)

        # arrow and formatted delta
        res["Direction"] = np.where(res["Δ Health Score (pts)"] > 0, "↑ improves", "↓ worsens")
        res["Δ Health Score (pts)"] = res["Δ Health Score (pts)"].map(lambda x: f"{x:+.1f}")

        st.dataframe(
            res[["Feature", "Current", "Alternative", "Direction", "Δ Health Score (pts)"]],
            use_container_width=True,
            hide_index=True
        )

except Exception as e:
    st.warning(f"Could not generate suggestions: {e}")

st.caption("⚠️ Counterfactual changes are hypothetical and for learning only.")

with st.container(border=True):
    st.markdown(
        "**Health Score (↑ better):** a linear transformation of the model’s expected obesity level into a 0–100 scale, where a higher score indicates a healthier predicted outcome."
    )
    # st.latex(r"\text{Severity}=\sum_{i=0}^{6} p_i\, i \quad\text{and}\quad \text{HealthScore}=100\!\left(1-\frac{\text{Severity}}{6}\right)")
    st.latex(r"\text{HealthScore}=100\!\left(1-\frac{\sum_{i=0}^{6} p_i\, i}{6}\right)")
