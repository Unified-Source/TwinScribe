@echo off
rem Development launcher for the command line, from the project's virtual environment.
rem   twinscribe.cmd                      opens the window
rem   twinscribe.cmd run <folder or file> transcribes, with progress in this console
rem   twinscribe.cmd check                reports the machine, the models and the plan
rem A portable copy uses the launchers that tools\build_portable.py writes instead.
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m twinscribe %*
) else (
  echo The project virtual environment was not found beside this file.
  echo Create it and install the package with its engines and app extras, or use tools\build_portable.py.
  exit /b 2
)
endlocal
