@echo off
title RF Spectrum System Verification Suite
echo =====================================================================
echo Running Master System Integration & Regression Test Suite...
echo =====================================================================
echo.
cd /d "%~dp0"
python scripts/run_full_system_verification.py
echo.
pause
