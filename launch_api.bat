@echo off
title RF Spectrum Monitoring - FastAPI REST Microservice
echo =====================================================================
echo Starting High-Throughput RF Spectrum REST Microservice
echo Host: http://127.0.0.1:8000
echo Interactive Swagger Docs : http://127.0.0.1:8000/docs
echo Interactive ReDoc        : http://127.0.0.1:8000/redoc
echo =====================================================================
echo.
cd /d "%~dp0"
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
pause
