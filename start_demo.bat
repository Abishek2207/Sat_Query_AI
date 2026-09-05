@echo off
echo ===================================================
echo     SATQUERY AI - OFFLINE DEMO LAUNCHER
echo ===================================================
echo.
echo Starting FastAPI Backend...
start cmd /k "cd backend && set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1 && uvicorn app.main:app --host 127.0.0.1 --port 8005"

echo Waiting for backend to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

echo Starting React Frontend...
start cmd /k "cd frontend && set VITE_API_BASE_URL=http://127.0.0.1:8005 && npm run dev"

echo.
echo ===================================================
echo ALL SYSTEMS GO! 
echo 1. The backend is running on http://127.0.0.1:8005
echo 2. The frontend is opening in your browser.
echo ===================================================
pause
