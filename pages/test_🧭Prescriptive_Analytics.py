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
    sec = st.radio(
        "Navigate on the page",
        ["Prescriptive Analytics", "SHAP Summary Plot", "SHAP Force Plot", "What-if Analysis"],
        index=0
    )
    mapping = {
        "Prescriptive Analytics": "Prescriptive Analytics",
        "SHAP Summary Plot": "SHAP Summary Plot",
        "SHAP Force Plot": "SHAP Force Plot",
        "What-if Analysis": "What-if Analysis",
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

# -------------------------------
# SHAP Summary Plot (same content; smaller fonts/figure)
# -------------------------------
st.markdown('<div id="SHAP Summary Plot"></div>', unsafe_allow_html=True)
st.subheader("SHAP Summary Plot")

try:
    sv_summary, _ = get_per_class_shap(shap_values, class_idx=int(prediction[0]))

    # compact layout
    plt.rcParams.update({
        "font.size": 6,                # smaller font overall
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })
    shap.summary_plot(sv_summary, user_data, show=False, plot_type="bar")
    fig = plt.gcf()
    fig.set_size_inches(6, 3)          # smaller figure
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=False)

except Exception as e:
    st.warning(f"SHAP summary plot failed: {e}")

# -------------------------------
# SHAP Force Plot — EXACTLY your original version
# -------------------------------
st.markdown('<div id="SHAP Force Plot"></div>', unsafe_allow_html=True)
st.subheader("SHAP Force Plot")

try:
    # keep your original indexing & API
    st_shap(
        shap.force_plot(
            explainer.expected_value[int(prediction[0])],
            shap_values[:, :, int(prediction[0])],
            user_data.iloc[0, :]
        ),
        height=200,
        width=1000
    )
except Exception as ee:
    st.warning(f"Force plot could not be rendered: {ee}")

# -------------------------------
# What-if / Counterfactual Analysis
# -------------------------------
st.markdown('<div id="What-if Analysis"></div>', unsafe_allow_html=True)
st.markdown("---")
st.subheader("What-if / Counterfactual Analysis")

st.caption(
    "Adjust feasible, encoded factors below to see how the predicted class distribution and a simple severity score change. "
    "Educational use only—this does **not** imply causality or medical advice."
)

CLASSES = [
    "Insufficient Weight", "Normal Weight", "Overweight Level I",
    "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III"
]

# Features we allow users to adjust (encoded values only)
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

def severity_score(p: np.ndarray) -> float:
    idx = np.arange(len(p))
    return float((p * idx).sum())

base_df   = user_data.copy()
base_prob = proba_row(loaded_model, base_df)
base_sev  = severity_score(base_prob)
base_cls  = CLASSES[int(np.argmax(base_prob))]

m1, m2 = st.columns(2)
with m1:
    st.metric("Predicted class", base_cls)
with m2:
    st.metric("Severity (expected class index ↓)", f"{base_sev:.2f}")

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
            cf_row[feat] = st.selectbox(feat, allowed, index=idx, key=f"cf_{feat}")

cf_df   = pd.DataFrame([cf_row])
cf_prob = proba_row(loaded_model, cf_df)
cf_sev  = severity_score(cf_prob)
delta   = cf_sev - base_sev

left, right = st.columns(2)
with left:
    st.write("**Adjusted class probabilities**")
    st.bar_chart(pd.Series(cf_prob, index=CLASSES))
with right:
    st.write("**Original vs Adjusted Severity**")
    fig_cmp, ax_cmp = plt.subplots(figsize=(4, 2))
    bars = ax_cmp.barh(["Original", "Adjusted"], [base_sev, cf_sev])
    ax_cmp.set_xlabel("Expected Class Index (lower is better)")
    ax_cmp.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()
    st.pyplot(fig_cmp, clear_figure=True)
    st.metric("Δ Severity (Adjusted − Original)", f"{delta:+.2f}")

st.caption("⚠️ Counterfactual changes are hypothetical and for learning only.")