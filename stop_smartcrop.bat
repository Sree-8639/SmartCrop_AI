@echo off
title SmartCrop AI - Stop Servers
color 0C

echo ==============================================================================
echo             SMARTCROP AI - SHUTTING DOWN SERVERS
echo ==============================================================================
echo.

echo Terminating processes on Port 8000 (FastAPI) and Port 8501 (Streamlit)...
powershell -Command "Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { (Get-NetTCPConnection -OwningProcess $_.Id -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in 8000, 8501 }) } | Stop-Process -Force -ErrorAction SilentlyContinue"

powershell -Command "Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"

echo.
echo All SmartCrop AI servers have been stopped successfully.
echo.
pause

