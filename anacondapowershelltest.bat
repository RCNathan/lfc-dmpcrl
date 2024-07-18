@echo off
setlocal
set ANACONDA_PATH_TEST=C:\Users\Natha\anaconda3
powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& '%ANACONDA_PATH_TEST%\shell\condabin\conda-hook.ps1' ; conda activate '%ANACONDA_PATH_TEST%' ; conda activate lfc-dmpcrl"
endlocal