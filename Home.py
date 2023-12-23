import streamlit as st

st.set_page_config(
    page_title="你好",
    page_icon="👋",
)

st.write("# 深圳出租车订单时间预测")


st.markdown(
    """
    Streamlit 是一个专为机器学习和数据科学项目而构建的开源应用框架。
    本程序利用Streamlit构建，尝试基于大数据机器学习对深圳出租车的订单时间做出预测。
    ### Streamlit官网&文档
    - 官网 [streamlit.io](https://streamlit.io)
    - [文档](https://docs.streamlit.io)
    
    ### 本程序使用的参考教程以及数据集
    -  教程[pygeo-tutorial](https://gitee.com/ni1o1/pygeo-tutorial)
    -  数据集[深圳出租车数据-Urban Data Release V2 Taxi GPS Data](https://www.cs.rutgers.edu/~dz220/data.html)
"""
)