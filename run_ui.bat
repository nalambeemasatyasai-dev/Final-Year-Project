@echo off
echo ========================================================
echo   Launching Student Engagement Recognition Streamlit UI
echo ========================================================
python -m streamlit run app.py --server.headless=false
pause
