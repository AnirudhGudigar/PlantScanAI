@echo off
title PlantScan AI - Backend API Server
echo ========================================================
echo PlantScan AI: Starting Production FastAPI Server
echo Swagger UI Docs: http://localhost:8000/docs
echo Health Check:    http://localhost:8000/api/v1/health
echo ========================================================
echo.
cd /d "%~dp0"
set PYTHONPATH=%CD%
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
