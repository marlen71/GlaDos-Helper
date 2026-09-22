@echo off
chcp 65001 >nul
echo === Установка GLaDOS Helper ===
where python >nul 2>nul || (echo Сначала установите Python 3.10+ с python.org && pause && exit /b 1)

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo.
echo Готово! Запуск: run.bat
pause
