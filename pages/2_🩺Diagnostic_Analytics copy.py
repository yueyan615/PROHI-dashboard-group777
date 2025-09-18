# app_heatmap.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ---------------------- 页面设置 ----------------------
st.set_page_config(
    # page_title="Obesity Dashboard",
    page_icon="./img/logo1.png",
    layout="wide"
)

# ---------------------- 读数据 ----------------------
# 优先读 parquet；若不存在则尝试 CSV（包括 /mnt/data 的备份）
parquet_path = './assets/ObesityDataSet_cleaned.parquet'
csv_candidates = [
    './assets/ObesityDataSet_cleaned.csv',
    '/mnt/data/ObesityDataSet_cleaned.csv',  # 你上传的文件常见路径
]

df = None
if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
else:
    for p in csv_candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

if df is None:
    st.error("未找到数据文件。请确认存在以下任一路径：\n"
             f"- {parquet_path}\n- " + "\n- ".join(csv_candidates))
    st.stop()

# ---------------------- 侧边栏（Logo） ----------------------
img1 = './img/logo_nb.png'
try:
    # 新版 Streamlit 提供 st.logo；若你的版本不支持可改为 st.sidebar.image
    st.logo(img1, size="large", icon_image=None)
except Exception:
    st.sidebar.image(img1, use_container_width=True)

# ---------------------- 主体标题与说明 ----------------------
st.markdown("# Diagnostic Analytics")
st.markdown("""
本节用于探索变量之间的关系。通过相关性热图，帮助你初步识别可能与肥胖水平有关的特征间联动。
""")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- 控件：方法 + 范围 ----------------------
col1, col2 = st.columns([1, 1])
with col1:
    method = st.selectbox("**Correlation method**", ["pearson", "spearman", "kendall"], index=0,
                          help="Pearson 适合线性关系；Spearman 更鲁棒；Kendall 最稳但更慢。")
with col2:
    scope = st.selectbox("**Columns to include**",
                         ["numeric only", "one-hot (all)"],
                         index=0,
                         help="numeric only：仅数值列；one-hot (all)：对非数值列做独热，再与数值列一起计算相关。")

# ---------------------- 构造用于相关性的矩阵 ----------------------
if scope == "numeric only":
    mat = df.select_dtypes(include=np.number).copy()
else:
    # 对所有非数值列做独热编码；更适合与 Pearson 搭配
    # 若选择 spearman/kendall，超大 0/1 矩阵会更慢，请酌情使用
    mat = pd.get_dummies(df, drop_first=False)

if mat.shape[1] < 2:
    st.warning("可用于计算相关性的列不足 2 个。请检查数据或更换“Columns to include”的选择。")
    st.stop()

# 大矩阵时，关闭格子内数字以避免重叠
show_text = mat.shape[1] <= 30

# ---------------------- 计算相关矩阵 ----------------------
corr = mat.corr(method=method)

# ---------------------- 绘图（热图） ----------------------
custom_scale = [
    (0.0, "#cf4846"),   # -1
    (0.5, "#ffffff"),   #  0
    (1.0, "#46cdcf"),   # +1
]

fig = px.imshow(
    corr,
    labels=dict(x="Feature", y="Feature", color="Correlation"),
    x=corr.columns,
    y=corr.index,
    color_continuous_scale=custom_scale,
    zmin=-1, zmax=1,
    text_auto=".2f" if show_text else False,
    title=f"Correlation matrix ({method}, {scope})"
)

# 增大注释（格子内数字）字体
if show_text:
    fig.update_traces(textfont={"size": 10})

# 画布与字体
fig.update_layout(
    width=1200,
    height=1000,
    margin=dict(l=150, r=50, t=80, b=150),
    title=dict(font=dict(size=20)),
    font=dict(size=12)
)

# 轴标签和刻度
fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
fig.update_yaxes(tickfont=dict(size=11))

# colorbar 设置
fig.update_coloraxes(
    colorbar=dict(
        thickness=20,
        lenmode="fraction",
        len=1.0,
        tickvals=[-1, 0, 1],
        ticktext=["-1", "0", "1"],
        title=dict(text=""),
        y=0.5,
        yanchor="middle",
    ),
    cmin=-1,
    cmax=1
)

st.plotly_chart(fig, use_container_width=False)

# ---------------------- 贴心提示 ----------------------
with st.expander("How to choose method/scope? (click to expand)", expanded=False):
    st.markdown("""
- **Pearson**: Most common; measures linear relationships; sensitive to outliers.  
- **Spearman**: Rank-based; better for monotonic but non-linear relationships.  
- **Kendall**: Concordance-based; more robust but slower to compute.  

- **numeric only**: Safest and easiest to interpret; avoids imposing artificial order on categorical features.  
- **one-hot (all)**: Converts categorical features to 0/1 dummies and computes correlations together with numeric columns; good for exploration but matrices can become large.  
""")
