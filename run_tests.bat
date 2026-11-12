@echo off
title RF Spectrum Monitoring - Pytest Regression Suite
echo =====================================================================
echo Running Automated Pytest Regression Test Suite
echo =====================================================================
echo.
cd /d "%~dp0"
python -m pytest -v tests/
echo.
echo =====================================================================
pause
