@echo off
echo ========================================================
echo   Launching Student Engagement Recognition Streamlit UI
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py --server.headless=false
pause
