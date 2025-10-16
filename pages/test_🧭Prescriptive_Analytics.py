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

# -------------------------------
# SHAP Summary Plot (same content; smaller fonts/figure)
# -------------------------------
st.markdown('<div id="SHAP Summary Plot"></div>', unsafe_allow_html=True)
st.subheader("SHAP Summary Plot")

try:
    sv_summary, _ = get_per_class_shap(shap_values, class_idx=int(prediction[0]))

    # compact layout
    plt.rcParams.update({
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })
    shap.summary_plot(sv_summary, user_data, show=False, plot_type="bar")
    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=False)

except Exception as e:
    st.warning(f"SHAP summary plot failed: {e}")

# -------------------------------
# SHAP Force Plot
# -------------------------------
st.markdown('<div id="SHAP Force Plot"></div>', unsafe_allow_html=True)
st.subheader("SHAP Force Plot")

try:
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

# -------------------------------
# Top-3 Suggestions
# -------------------------------
st.markdown('<div id="Top-3 Suggestions"></div>', unsafe_allow_html=True)
st.markdown("### Top-3 Single-Step Suggestions")

st.info(
    r"""
We try **one change at a time** on actionable features and pick the **top-3** that **lower the Severity score** the most.  
Lower Δ means better (↓ severity).
"""
)

try:
    base_sev = severity_score(base_prob)
    current  = base_df.iloc[0].copy()
    suggestions = []

    for feat, allowed in ACTIONABLE_BOUNDS.items():
        if feat not in current.index:
            continue
        cur_val = int(current[feat])
        for alt in allowed:
            if int(alt) == cur_val:
                continue
            test = current.copy()
            test[feat] = int(alt)
            new_prob = proba_row(loaded_model, pd.DataFrame([test]))
            new_sev  = severity_score(new_prob)
            delta    = float(new_sev - base_sev)
            suggestions.append((feat, cur_val, int(alt), delta))

    if len(suggestions) == 0:
        st.info("No actionable single-step suggestions available.")
    else:
        res = (
            pd.DataFrame(
                suggestions, 
                columns=["Feature", "Current", "Alternative", "Δ Severity"]
            )
            .sort_values("Δ Severity", ascending=True)
            .head(3)
            .reset_index(drop=True)
        )
        res["Direction"]   = np.where(res["Δ Severity"] < 0, "↓ improves", "↑ worsens")
        res["Δ Severity"]  = res["Δ Severity"].map(lambda x: f"{x:+.2f}")

        st.dataframe(
            res[["Feature", "Current", "Alternative", "Direction", "Δ Severity"]],
            use_container_width=True,
            hide_index=True
        )

except Exception as e:
    st.warning(f"Could not generate suggestions: {e}")

st.caption("⚠️ Counterfactual changes are hypothetical and for learning only.")

with st.container(border=True):
    st.markdown(
        "**Severity (↓ better):** expected class index. "
        "Classes 0–6 = Insufficient Weight, …, Obesity Type III."
    )
    st.latex(r"\text{Severity}=\sum_{i=0}^{6} p_i\, i \quad\text{where } p_i=\Pr(\text{class}=i)")