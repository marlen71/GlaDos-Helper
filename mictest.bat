@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>nul

set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" goto novenv

"%VPY%" "%~dp0mic_test.py"
echo.
pause
exit /b 0

:novenv
echo.
echo [ERROR] Virtual environment not found.
echo Please run install.bat first.
echo.
pause
exit /b 1
