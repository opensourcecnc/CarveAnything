@echo off
powershell -ExecutionPolicy Bypass -NoExit -Command "& '%~dp0venv\Scripts\Activate.ps1'; python CarveAnything.py"
pause