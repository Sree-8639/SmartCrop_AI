@echo off
title SmartCrop AI - Platform Launcher
color 0A

echo ==============================================================================
echo             SMARTCROP AI - AUTONOMOUS AG-PERCEPTION LAUNCHER
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking active server instances...
powershell -Command "Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { (Get-NetTCPConnection -OwningProcess $_.Id -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in 8000, 8501 }) } | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>&1

echo [2/3] Starting FastAPI Inference Backend (Port 8000)...
start "SmartCrop AI - Backend (Port 8000)" /min cmd /c "python -m uvicorn app.api.routes:app --host 127.0.0.1 --port 8000"

echo     Waiting for backend neural models to initialize...
powershell -Command "for ($i=0; $i -lt 30; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Backend is taking longer than expected. Continuing to launch frontend...
) else (
    echo     FastAPI Backend is ONLINE and healthy!
)

echo.
echo [3/3] Starting Streamlit Interactive Frontend (Port 8501)...
start "SmartCrop AI - Frontend (Port 8501)" cmd /c "python -m streamlit run app/main.py --server.port 8501 --server.address 127.0.0.1"

echo     Opening web browser at http://localhost:8501 ...
powershell -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:8501'"

echo.
echo ==============================================================================
echo  SMARTCROP AI IS RUNNING!
echo.
echo  - Frontend Web UI : http://localhost:8501
echo  - Backend API     : http://127.0.0.1:8000/api/health
echo  - API Docs (Docs) : http://127.0.0.1:8000/docs
echo.
echo  Keep the server windows open while using the application.
echo  To stop the application, double-click 'stop_smartcrop.bat'.
echo ==============================================================================
pause
