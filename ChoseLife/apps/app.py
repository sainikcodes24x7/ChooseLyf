import heart
import kidney
import exercise
import streamlit as st


PAGES = {
    "Heart Diesease Detection": heart,
    "Kidney Diesease Detection": kidney,
    "Exercise": exercise
}
st.sidebar.title('NAVIGATION')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()