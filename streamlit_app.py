import streamlit as st
x = st.slider("Select a value")
st.write(x, "squared is", x * x)

$ streamlit run streamlit_app.py
