@echo off
rem Development launcher: opens the application window from the project's virtual environment.
rem A portable copy uses the launchers that tools\build_portable.py writes instead.
setlocal
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" -m twinscribe.app %*
) else (
  echo The project virtual environment was not found beside this file.
  echo Create it and install the package with its app extra, or use tools\build_portable.py.
)
endlocal
