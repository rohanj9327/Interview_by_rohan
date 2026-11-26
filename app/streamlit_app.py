"""Streamlit entry point:  uv run streamlit run app/streamlit_app.py"""

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))  # make ui_common importable from views

st.set_page_config(page_title="AI Interview Simulator", page_icon="🎙️", layout="wide")
logging.basicConfig(level=logging.WARNING)

pages = st.navigation([
    st.Page("views/interview.py", title="Interview", icon="🎙️", default=True),
    st.Page("views/history.py", title="History", icon="📈"),
])
pages.run()
