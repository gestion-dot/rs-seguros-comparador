@echo off
cd /d "%~dp0"
echo Iniciando RS Seguros Comparador...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
