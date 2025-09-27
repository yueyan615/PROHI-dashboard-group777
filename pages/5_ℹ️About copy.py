import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="radio 导航滚动", layout="wide")

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
    sec = st.radio("跳转到", ["简介", "数据探索", "结论"], index=0)
    # 放在选择控件后立刻触发滚动
    mapping = {"简介": "sec-intro", "数据探索": "sec-explore", "结论": "sec-conclusion"}
    scroll_to(mapping[sec])

st.title("radio 导航滚动")

st.markdown('<div id="sec-intro"></div>', unsafe_allow_html=True)
st.header("简介")
st.write("..." * 30)

st.markdown('<div id="sec-explore"></div>', unsafe_allow_html=True)
st.header("数据探索")
st.write("..." * 30)

st.markdown('<div id="sec-conclusion"></div>', unsafe_allow_html=True)
st.header("结论")
st.write("..." * 30)
