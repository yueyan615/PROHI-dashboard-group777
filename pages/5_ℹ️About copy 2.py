import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="侧边栏：页间 + 页内导航", layout="wide")

# ========= 工具 =========
def scroll_to(element_id: str):
    # 在本次脚本 run 结束时，把父文档滚动到指定元素
    components.html(
        f"""
        <script>
        const target = window.parent.document.getElementById("{element_id}");
        if (target) {{
            target.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
        """,
        height=0,
    )

def anchor(id_: str):
    # 在需要定位的地方插一个锚点
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

# ========= 侧边栏 =========
with st.sidebar:
    st.markdown("### 页面导航")  # ——“页间”导航区——
    # 这里根据你的实际页面结构填写。主页通常是主脚本名（例如 Home.py）
    st.page_link("Welcome.py", label="🏠 首页")
    st.page_link("pages/1_📊Descriptive_Analytics.py", label="📊 数据分析")
    st.page_link("pages/2_🩺Diagnostic_Analytics.py", label="⚙️ 设置")
    # 也可以放外链：
    # st.page_link("https://example.com", label="🧭 文档")

    st.divider()

    st.markdown("### 本页目录")  # ——“页内”导航区——
    toc = [
        ("快速开始", "sec-quickstart"),
        ("可视化",   "sec-viz"),
        ("常见问题", "sec-faq"),
    ]
    # 用 radio/下拉都行；radio 切换时会重跑，本段 JS 会在本次 run 结束后执行滚动
    sec_label = st.radio("On this page: ", [label for label, _ in toc], label_visibility="collapsed")
    id_map = {label: id_ for label, id_ in toc}
    scroll_to(id_map[sec_label])

# ========= 正文 =========
st.title("示例页：同时包含页间与页内导航")

anchor("sec-quickstart")
st.header("快速开始")
st.write("这里写内容……" * 20)

anchor("sec-viz")
st.header("可视化")
st.write("这里放图表/表格……" * 20)

anchor("sec-faq")
st.header("常见问题")
st.write("这里是 FAQ ……" * 20)
