#!/bin/bash

# Log file
LOG_FILE="script_errors.log"
echo "Starting script execution..." > "$LOG_FILE"

# Function to run a script and log failures
run_script() {
    script_name=$1
    echo "Running $script_name..."
    
    python3 "$script_name"
    if [ $? -ne 0 ]; then
        echo "$script_name failed!" | tee -a "$LOG_FILE"
    else
        echo "$script_name completed successfully!" >> "$LOG_FILE"
    fi
}

# Run scripts sequentially
run_script dmpcrl_evaluate.py # includes both mpcrl and dmpcrl
run_script sc_mpc_cmd.py 

echo "All scripts attempted. Check $LOG_FILE for any failures."
