import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Diagnostic | Obesity Analytics",
    page_icon="./img/logo1.png",
    layout="wide"
)

# load the dataset
file_name = './assets/ObesityDataSet_BMI.parquet'
df = pd.read_parquet(file_name)

############################ SIDEBAR
### Logo
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
    sec = st.radio("Navigate on the page", ["Diagnostic Analytics","Heatmap"], index=0)
    mapping = {"Diagnostic Analytics": "Diagnostic Analytics", "Heatmap": "Heatmap"}
    scroll_to(mapping[sec])



############################ MAIN BODY
st.markdown('<div id="Diagnostic Analytics"></div>', unsafe_allow_html=True)
""" # Diagnostic Analytics"""

"""
This section helps users explore relationships between variables. Through correlations, statistical tests, and clustering, users can investigate potential factors that explain differences in obesity levels.
"""



########################### 1
st.markdown('<div id="Heatmap"></div>', unsafe_allow_html=True)
"""## Heatmap of Correlation Matrix"""

# 让用户选择相关系数方法
method = st.selectbox("**Correlation method**", ["pearson", "spearman", "kendall"], index=0)

# 把所有特征编码为数值（数值保留，category->cat.codes，bool->int，其他用 factorize）
enc = pd.DataFrame(index=df.index)
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        enc[col] = s
    elif s.dtype.name == "category":
        enc[col] = s.cat.codes
    elif pd.api.types.is_bool_dtype(s):
        enc[col] = s.astype(int)
    else:
        # object / other -> 整数编码（缺失为 -1）
        enc[col] = pd.factorize(s, sort=False)[0]

# 计算相关矩阵
corr = enc.corr(method=method)

custom_scale = [
    (0.0, "#0072b2"),   # 对应 zmin (-1)
    (0.5, "#ffffff"),   # 对应 0
    (1.0, "#e69f00"),   # 对应 zmax (1)
]

fig = px.imshow(
    corr,
    labels=dict(x="Feature", y="Feature", color="Correlation"),
    x=corr.columns,
    y=corr.index,
    color_continuous_scale=custom_scale,
    zmin=-1, zmax=1,
    text_auto=".2f",
    title=f"Correlation matrix ({method})"
)

# 增大注释（格子内数字）字体
fig.update_traces(textfont={"size": 10})

# 放大画布并调整边距，增加整体字体（坐标轴/标题）
fig.update_layout(
    width=900,
    height=700,
    margin=dict(l=50, r=50, t=80, b=50),
    title=dict(font=dict(size=20)),
    font=dict(size=12)
)

# 轴标签和刻度更清晰
fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
fig.update_yaxes(tickfont=dict(size=11))

# 调整 colorbar 宽度与长度
fig.update_coloraxes(
    colorbar=dict(
        thickness=20,
        lenmode="fraction",
        len=1.0,                  # 占满热图高度
        tickvals=[-1, 0, 1],
        ticktext=["-1", "0", "1"],
        title=dict(text=""),     # 不显示标题
        y=0.5,
        yanchor="middle",
        x=0.9,          
        xanchor="left",
        xpad=0.2
    ),
    cmin=-1,
    cmax=1
)
# 在 Streamlit 中保持自定义宽高（use_container_width=False）
st.plotly_chart(fig, use_container_width=False)


########################### 2
# ...existing code...

from scipy.stats import chi2_contingency

# Select nominal variables (treat bool as nominal as well)
nominal_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
if len(nominal_cols) < 2:
    st.info("There are not enough nominal variables to plot a heatmap.")
else:
    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        if ct.size == 0:
            return 0.0
        chi2 = chi2_contingency(ct, correction=False)[0]
        n = ct.to_numpy().sum()
        r, k = ct.shape
        denom = n * (min(r - 1, k - 1))
        return float(0.0 if denom == 0 else np.sqrt(chi2 / denom))

    n = len(nominal_cols)
    mat = pd.DataFrame(np.zeros((n, n)), index=nominal_cols, columns=nominal_cols, dtype=float)

    for i, a in enumerate(nominal_cols):
        for j, b in enumerate(nominal_cols):
            if i > j:
                mat.iloc[i, j] = mat.iloc[j, i]
                continue
            if a == b:
                mat.iloc[i, j] = 1.0
                continue
            try:
                val = cramers_v(df[a], df[b])
            except Exception:
                val = 0.0
            mat.iloc[i, j] = val
            mat.iloc[j, i] = val

    # 绘图：0..1，从白到 #46cdcf
    custom_scale = [(0.0, custom_scale[2][1]), (1.0, custom_scale[0][1])]

    fig = px.imshow(
        mat,
        labels=dict(x="Feature", y="Feature", color="Cramér's V"),
        x=mat.columns,
        y=mat.index,
        color_continuous_scale=custom_scale,
        zmin=0, zmax=1,
        text_auto=".2f",
        title="Cramér's V (nominal vs nominal)"
    )

    fig.update_traces(textfont={"size": 12})
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        title=dict(font=dict(size=18)),
        font=dict(size=12)
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_coloraxes(
        colorbar=dict(
            thickness=20,
            lenmode="fraction",
            len=1.0,
            tickvals=[0, 0.5, 1],
            ticktext=["0", "0.5", "1"],
            title=dict(text=""),
            y=0.5,
            yanchor="middle",
        ),
        cmin=0,
        cmax=1
    )

    # st.plotly_chart(fig, use_container_width=False)


#################################### 3

# st.markdown("## The association strength of each column with Obesity level")
# st.write("methods: Cramér's V for nominal↔nominal; eta (correlation ratio) for nominal→numeric")
# target = "Obesity_level"
# if target not in df.columns:
#     st.error(f"{target} not found")
# else:
#     # 辅助：Cramér's V（名义-名义）
#     def cramers_v(x, y):
#         ct = pd.crosstab(x.fillna("##NA##"), y.fillna("##NA##"))
#         if ct.size == 0:
#             return 0.0
#         chi2 = chi2_contingency(ct, correction=False)[0]
#         n = ct.to_numpy().sum()
#         r, k = ct.shape
#         denom = n * (min(r - 1, k - 1))
#         return float(0.0 if denom == 0 else np.sqrt(chi2 / denom))

#     # 辅助：correlation ratio / eta（名义目标 - 数值特征）
#     def correlation_ratio(categories, measurements):
#         cat = np.asarray(categories)
#         meas = np.asarray(measurements, dtype=float)
#         mask = ~pd.isnull(cat) & ~pd.isnull(meas)
#         if mask.sum() == 0:
#             return 0.0
#         cat = cat[mask]
#         meas = meas[mask]
#         groups = {}
#         for c, m in zip(cat, meas):
#             groups.setdefault(c, []).append(m)
#         grand_mean = meas.mean()
#         ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
#         ss_total = ((meas - grand_mean) ** 2).sum()
#         return float(0.0 if ss_total == 0 else ss_between / ss_total)

#     rows = []
#     for col in df.columns:
#         if col == target:
#             continue
#         s = df[col]
#         # 将布尔视为名义
#         if pd.api.types.is_numeric_dtype(s):
#             # 数值特征：用 correlation ratio（Obesity_level 为分组）
#             val = correlation_ratio(df[target].astype(str), s)
#             method = "eta (nominal→numeric)"
#         else:
#             # 名义特征（包括 bool/object/category）：用 Cramér's V
#             val = cramers_v(df[col].astype(str), df[target].astype(str))
#             method = "Cramér's V (nominal↔nominal)"
#         rows.append({"Feature": col, "Association": float(val), "Method": method})

#     assoc_df = pd.DataFrame(rows).sort_values("Association", ascending=False).reset_index(drop=True)

#     # 显示表格（前 50）
#     st.subheader("Sorted by the strength of association with Obesity_level (0-1)")
#     st.dataframe(assoc_df, use_container_width=True)

#     # 绘图：条形图
#     fig_assoc = px.bar(
#         assoc_df,
#         x="Association",
#         y="Feature",
#         orientation="h",
#         color="Method",
#         color_discrete_map={
#             "Cramér's V (nominal↔nominal)": "#46cdcf",
#             "eta (nominal→numeric)": "#cf4846"
#         },
#         hover_data={"Association": ":.3f"},
#         title="Feature vs Obesity_level — Association strength",
#         text=assoc_df["Association"].map(lambda v: f"{v:.2f}")
#     )
#     fig_assoc.update_layout(
#         height= max(400, 40 * len(assoc_df)), 
#         margin=dict(l=300, r=50, t=60, b=50),
#         yaxis=dict(autorange="reversed"),
#         showlegend=True
#     )
#     fig_assoc.update_traces(textposition="outside", textfont=dict(size=11))
#     st.plotly_chart(fig_assoc, use_container_width=False)

