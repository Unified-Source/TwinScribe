@echo off
rem Development launcher for the window without a console, from the project's virtual
rem environment. Recordings or folders given as arguments are added to the library.
rem A portable copy uses the launchers that tools\build_portable.py writes instead.
setlocal
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" -m twinscribe.app %*
) else (
  echo The project virtual environment was not found beside this file.
  echo Create it and install the package with its engines and app extras, or use tools\build_portable.py.
  exit /b 2
)
endlocal
