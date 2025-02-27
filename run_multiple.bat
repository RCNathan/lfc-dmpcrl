@echo off
setlocal enabledelayedexpansion

:: Log file
set LOG_FILE=script_errors.log
echo Starting script execution... > %LOG_FILE%

:: Function to run a script and log failures
for %%S in (dmpcrl_evaluate.py sc_mpc_cmd.py) do (
    echo Running %%S...
    python "%%S"
    
    if errorlevel 1 (
        echo %%S failed! >> %LOG_FILE%
        echo %%S failed!
    ) else (
        echo %%S completed successfully! >> %LOG_FILE%
    )
)

echo All scripts attempted. Check %LOG_FILE% for any failures.
