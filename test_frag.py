import streamlit as st
import time

if "count" not in st.session_state:
    st.session_state.count = 0

@st.fragment
def toggle_frag():
    st.toggle("Show", key="show_vid")
    st.write("Toggle is", st.session_state.get("show_vid"))

toggle_frag()

placeholder = st.empty()

if st.button("Start"):
    for i in range(20):
        if st.session_state.get("show_vid"):
            placeholder.write(f"Running {i}")
        else:
            placeholder.write("Hidden")
        time.sleep(1)
        st.session_state.count += 1
