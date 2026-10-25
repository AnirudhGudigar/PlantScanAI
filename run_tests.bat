@echo off
title PlantScan AI - Run Unit Tests
echo ========================================================
echo PlantScan AI: Running Automated Pytest Unit Test Suite
echo ========================================================
echo.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest backend/tests -v
echo.
echo ========================================================
echo Tests Completed!
echo ========================================================
pause
