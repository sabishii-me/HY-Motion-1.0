@echo off
REM HY-Motion Profile Launcher

cd /d "%~dp0"

:menu
cls
echo ========================================
echo HY-Motion 1.0 Profile Launcher
echo ========================================
echo.
echo Available Profiles:
echo.
echo [1] RTX 5080 INT4 (6-8GB VRAM, Fast, Recommended)
echo [2] RTX 5080 INT8 (12-13GB VRAM, Higher Quality)
echo [3] RTX 5080 Official (Full Precision, CPU Offload, Slower)
echo.
echo [Q] Quit
echo.
echo ========================================

choice /C 123Q /N /M "Select profile: "

if errorlevel 4 goto :eof
if errorlevel 3 goto :official
if errorlevel 2 goto :int8
if errorlevel 1 goto :int4

:int4
cls
echo ========================================
echo RTX 5080 INT4 Profile
echo ========================================
echo.
echo [1] Setup (first time only)
echo [2] Run Gradio App
echo [B] Back to menu
echo.
choice /C 12B /N /M "Select action: "

if errorlevel 3 goto :menu
if errorlevel 2 (
    cd profiles\rtx5080_int4
    call run_gradio.bat
    cd ..\..
    goto :menu
)
if errorlevel 1 (
    cd profiles\rtx5080_int4
    call setup.bat
    cd ..\..
    goto :menu
)

:int8
cls
echo ========================================
echo RTX 5080 INT8 Profile
echo ========================================
echo.
echo [1] Setup (first time only)
echo [2] Run Gradio App
echo [B] Back to menu
echo.
choice /C 12B /N /M "Select action: "

if errorlevel 3 goto :menu
if errorlevel 2 (
    cd profiles\rtx5080_int8
    call run_gradio.bat
    cd ..\..
    goto :menu
)
if errorlevel 1 (
    cd profiles\rtx5080_int8
    call setup.bat
    cd ..\..
    goto :menu
)

:official
cls
echo ========================================
echo RTX 5080 Official Profile
echo ========================================
echo.
echo [1] Setup (first time only)
echo [2] Run Gradio App
echo [B] Back to menu
echo.
choice /C 12B /N /M "Select action: "

if errorlevel 3 goto :menu
if errorlevel 2 (
    cd profiles\rtx5080_official
    call run_gradio.bat
    cd ..\..
    goto :menu
)
if errorlevel 1 (
    cd profiles\rtx5080_official
    call setup.bat
    cd ..\..
    goto :menu
)
