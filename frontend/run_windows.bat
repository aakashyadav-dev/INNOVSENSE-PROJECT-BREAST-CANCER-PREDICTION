@echo off
echo ========================================
echo Starting InnovSense Breast Cancer Prediction System
echo ========================================

REM Start Backend
echo Starting Backend Server...
cd backend
call venv\Scripts\activate
start cmd /k "uvicorn app.main:app --reload --port 8000"

REM Wait a bit
timeout /t 5

REM Start Frontend
echo Starting Frontend...
cd ..\frontend
call venv\Scripts\activate
start cmd /k "streamlit run streamlit_app.py"

echo.
echo ========================================
echo InnovSense is starting!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo ========================================
pause