@echo off
echo ======================================================================
echo  Neuro-Adaptive Financial Market Signal Analysis Web Application
echo  PBL Data Science College Project
echo ======================================================================
echo.
echo Starting Streamlit Web Dashboard...
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
