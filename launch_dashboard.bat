@echo off
title AI-Driven RF Spectrum Monitoring Dashboard
echo =====================================================================
echo Launching AI-Driven RF Spectrum Monitoring System Web Console...
echo =====================================================================
echo.
cd /d "%~dp0"
python -m streamlit run dashboard/app.py
pause
