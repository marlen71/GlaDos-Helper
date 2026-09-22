@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo [ОШИБКА] Виртуальное окружение не найдено: %VPY%
  echo Сначала запустите install.bat
  pause & exit /b 1
)

"%VPY%" "%~dp0main.py" %*
if errorlevel 1 pause
