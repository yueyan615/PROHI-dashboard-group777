import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="About | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

############################ SIDEBAR
img1 = './img/logo.svg'
st.logo(img1, size= "large", icon_image=None)  


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
    sec = st.radio("On this page", ["Dataset", "Team & Contact", "References"], index=0)
    mapping = {"Dataset": "Dataset", "Team & Contact": "Team & Contact", "References": "References"}
    scroll_to(mapping[sec])



############################ MAIN BODY
st.title("About")

# ===== Problem Description =====
st.markdown(
    "Obesity-Vision is a Streamlit dashboard built on the UCI obesity dataset. It integrates descriptive, diagnostic, predictive, and SHAP-based prescriptive analytics to turn lifestyle and physical-condition data into clear insights on obesity risk. The dashboard serves the public (health literacy), healthcare providers (consultation visuals), and public-health organizations (outreach), offering engaging, responsible, non-clinical guidance that supports prevention."
)

st.divider()

# ===== Dataset =====
st.markdown('<div id="Dataset"></div>', unsafe_allow_html=True)
st.markdown("## Dataset")
st.markdown(
    'We use the UCI dataset ["Estimation of Obesity Levels Based on Eating Habits and Physical Condition"]'
    "(https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). "
    "The original resource records lifestyle and physical-condition factors with the goal of classifying obesity levels. "
    "To address class imbalance, the dataset is widely distributed in a SMOTE-balanced form using Weka; 77% of the final records are synthetic. "
    "A side effect of this SMOTE pipeline is that some categorical fields end up stored as floating-point values instead of integer codes.\n\n"
    "For this dashboard, we load a cleaned, rounded dataset derived from that SMOTE-balanced form. During preprocessing we rounded and cast categorical variables "
    "to integer codes so category options are integers and better reflect their real-world meaning. Insights are intended for educational and exploratory use."
)

with st.expander("UCI dataset details"):
    st.markdown(
        "- **Size**: 2,111 records; 17 attributes (16 features + 1 target)\n"
        "- **Collection and preprocessing**: initial web survey of 485 responses (ages 14–61) over ~30 days; preprocessing included removal of missing/atypical data and normalization\n"
        "- **Regions**: Mexico, Peru, Colombia (Latin America)\n"
        "- **Label (target)**: seven-class obesity level — Insufficient Weight, Normal Weight, Overweight I, Overweight II, Obesity I, Obesity II, Obesity III; labels derived from BMI using thresholds aligned with WHO and Mexican Normativity\n"
        "  - Underweight < 18.5; Normal 18.5–24.9; Overweight 25.0–29.9; Obesity I 30.0–34.9; Obesity II 35.0–39.9; Obesity III ≥ 40\n"
        "- **Features (examples)**: high-calorie food frequency (FAVC), vegetables frequency (FCVC), number of main meals (NCP), snacking (CAEC), daily water intake (CH2O), alcohol consumption (CALC), calorie monitoring (SCC), physical activity (FAF), screen time (TUE), transport mode (MTRANS), plus gender, age, height, weight\n"
        "- **Imbalance handling**: SMOTE (Weka) used to address class imbalance; ~77% of the final 2,111 records are synthetic"
    )


st.divider()

# ===== Team & Contact =====
st.markdown('<div id="Team & Contact"></div>', unsafe_allow_html=True)
st.markdown("## Team & Contact")
st.markdown(
    "This dashboard was created by Group 777 as part of the assignment for Project Management and Tools for Health Informatics (PROHI).\n\n"
    "Team members:\n"
    "- Weiqi Kong\n"
    "- Yueyan Li\n"
    "- Zsolt Fehér\n"
    "- Christoffer Brändefors\n"
    "- Naznin Akhtar\n\n"
    "Contact:\n"
    "- Email: group777@su.se\n"
    "- Address: Stockholm University, SE-106 91 Stockholm, Sweden"
)

st.divider()

# ===== References =====
st.markdown('<div id="References"></div>', unsafe_allow_html=True)
st.markdown("## References")
st.markdown(
    "[1] Palechor FM, Manotas AH. Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. "
    "Data Brief. 2019;25:104344. DOI: 10.1016/j.dib.2019.104344.\n\n"
    "[2] UCI Machine Learning Repository. Estimation of Obesity Levels Based on Eating Habits and Physical Condition [Internet]. 2019. "
    "Available from: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition\n\n"
    "[3] World Health Organization. Obesity and overweight. [Internet]. 2021. "
    "Available from: https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight\n\n"
    "[4] Solomon DD, Khan S, Garg S, Gupta G, Almjally A, Alabduallah BI, et al. Hybrid Majority Voting: Prediction and Classification Model for Obesity. "
    "Diagnostics (Basel). 2023;13(15). DOI: 10.3390/diagnostics13152610."
)

# st.divider()
# st.caption("Scope: educational and exploratory use; not medical advice.")
