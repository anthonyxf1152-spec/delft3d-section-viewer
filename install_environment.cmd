@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "VIEWER_CONDA="
where conda >nul 2>&1
if not errorlevel 1 set "VIEWER_CONDA=conda"
if not defined VIEWER_CONDA if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "VIEWER_CONDA=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined VIEWER_CONDA if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "VIEWER_CONDA=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined VIEWER_CONDA if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "VIEWER_CONDA=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined VIEWER_CONDA if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "VIEWER_CONDA=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined VIEWER_CONDA (
  echo 请先安装 Miniconda 或 Anaconda，然后从 Anaconda Prompt 运行本脚本。
  pause
  exit /b 1
)
echo 正在创建 delft3d 环境，需要联网下载依赖。
echo 如果同名环境已经存在，请按 README 使用 conda env update 命令。
call "%VIEWER_CONDA%" env create --file "%~dp0environment.yml"
if errorlevel 1 (
  echo 环境创建失败，请查看上面的错误信息。
  pause
  exit /b 1
)
call "%VIEWER_CONDA%" run --no-capture-output -n delft3d python "%~dp0check_environment.py"
if errorlevel 1 (
  echo 依赖检查失败，请查看上面的错误信息。
  pause
  exit /b 1
)
echo 安装完成。双击 start.cmd 启动程序。
pause
exit /b 0
