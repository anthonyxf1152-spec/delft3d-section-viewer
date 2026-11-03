@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "APP_ENTRY=%~dp0outputs\time_stratigraphy_preview\run_interactive.py"
if not exist "%APP_ENTRY%" (
  echo 未找到程序入口：%APP_ENTRY%
  echo 请先完整解压代码，再运行此文件。
  pause
  exit /b 1
)
if defined DELFT3D_PYTHON (
  if not exist "%DELFT3D_PYTHON%" (
    echo DELFT3D_PYTHON 指定的解释器不存在：%DELFT3D_PYTHON%
    pause
    exit /b 1
  )
  set "VIEWER_PYTHON=%DELFT3D_PYTHON%"
  goto run_python
)
if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import numpy,scipy,matplotlib,cigvis,vispy,PySide6" >nul 2>&1
    if not errorlevel 1 (
      set "VIEWER_PYTHON=%CONDA_PREFIX%\python.exe"
      goto run_python
    )
  )
)
set "VIEWER_CONDA="
where conda >nul 2>&1
if not errorlevel 1 set "VIEWER_CONDA=conda"
if not defined VIEWER_CONDA if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "VIEWER_CONDA=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined VIEWER_CONDA if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "VIEWER_CONDA=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined VIEWER_CONDA if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "VIEWER_CONDA=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined VIEWER_CONDA if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "VIEWER_CONDA=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined VIEWER_CONDA (
  echo 未找到 Conda。请在 Anaconda Prompt 中执行 conda activate delft3d 后运行 start.cmd。
  echo 也可将 DELFT3D_PYTHON 设置为该环境 python.exe 的完整路径。
  pause
  exit /b 1
)
call "%VIEWER_CONDA%" run --no-capture-output -n delft3d python -u "%APP_ENTRY%" %*
goto finished

:run_python
"%VIEWER_PYTHON%" -u "%APP_ENTRY%" %*

:finished
set "VIEWER_EXIT=%ERRORLEVEL%"
if not "%VIEWER_EXIT%"=="0" pause
exit /b %VIEWER_EXIT%
