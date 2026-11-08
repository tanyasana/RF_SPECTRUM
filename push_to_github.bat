@echo off
title Push RF Spectrum Monitoring System to GitHub
echo =====================================================================
echo Pushing Project to https://github.com/tanyasana/RF_SPECTRUM
echo =====================================================================
echo.
cd /d "%~dp0"

echo Current git status:
git status
echo.

set /p GITHUB_TOKEN="Enter your GitHub Personal Access Token (or press Enter if already logged in): "

if "%GITHUB_TOKEN%"=="" (
    echo Pushing using configured git credentials...
    git push origin main
) else (
    echo Pushing using provided Personal Access Token...
    git push https://%GITHUB_TOKEN%@github.com/tanyasana/RF_SPECTRUM.git main
)

echo.
echo =====================================================================
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Project successfully pushed to https://github.com/tanyasana/RF_SPECTRUM !
) else (
    echo [NOTICE] Push encountered an authentication error. Please ensure your token has 'repo' write permissions.
)
echo =====================================================================
pause
