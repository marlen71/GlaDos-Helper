@echo off
chcp 65001 >nul
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0main.py" %*
pause
