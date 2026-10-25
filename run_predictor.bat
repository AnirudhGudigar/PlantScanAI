@echo off
title PlantScan AI - CLI Predictor
echo ========================================================
echo PlantScan AI: Multi-Crop Disease Predictor CLI
echo ========================================================
echo.
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: run_predictor.bat ^<path_to_image^> [crop_name]
    echo Example: run_predictor.bat sample.jpg tomato
    echo.
    echo Defaulting to testing Treatment Database...
    ".venv\Scripts\python.exe" -c "from treatment_database.repository import TreatmentRepository; repo = TreatmentRepository(); print('Crops:', repo.list_supported_crops())"
) else (
    set CROP=%~2
    if "%CROP%"=="" set CROP=tomato
    ".venv\Scripts\python.exe" ml/inference/crop_predictor.py --image "%~1" --crop %CROP% --use-tflite
)
echo.
pause
