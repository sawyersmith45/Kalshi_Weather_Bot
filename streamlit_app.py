#!/usr/bin/env python3
"""Streamlit Cloud entrypoint that delegates to public dashboard app."""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "public_dashboard" / "streamlit_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
