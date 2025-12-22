#!/usr/bin/env bash
#
# run_experiments.sh - Automated experimental validation script
#
# This script runs the required experiments to prove ≥60% accuracy
# across multiple subjects as specified in Issue #14.
#
# Usage: bash run_experiments.sh
#

set -e  # Exit on error

echo "============================================================"
echo "  Total Perspective Vortex - Experimental Validation"
echo "  Issue #14: Proving ≥60% Accuracy Threshold"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the project root
if [ ! -d "src" ] || [ ! -f "src/mybci.py" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Create output directory for results
mkdir -p results
mkdir -p plots

# Log file
LOG_FILE="results/experiment_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Logging output to: $LOG_FILE"
echo ""

# Function to run training and capture results
run_experiment() {
    local subject=$1
    local run=$2

    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Running: Subject $subject, Run $run${NC}"
    echo -e "${YELLOW}======================================${NC}"
    echo ""

    # Run training
    echo "Training model with 5-fold cross-validation..."
    cd src
    python mybci.py $subject $run train --cv 5 2>&1 | tee -a "../$LOG_FILE"

    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo -e "${RED}Training failed for Subject $subject, Run $run${NC}" | tee -a "../$LOG_FILE"
        cd ..
        return 1
    fi

    # Run prediction
    echo ""
    echo "Running prediction on test set..."
    python mybci.py $subject $run predict 2>&1 | tee -a "../$LOG_FILE"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Prediction failed for Subject $subject, Run $run${NC}" | tee -a "../$LOG_FILE"
        cd ..
        return 1
    fi

    cd ..
    echo ""
    echo -e "${GREEN}✓ Completed: Subject $subject, Run $run${NC}"
    echo ""

    return 0
}

# Array to track results
declare -a results
declare -a subjects=(1 4 10)
declare -a runs=(6 14 6)

# Run experiments for each subject
success_count=0
total_count=3

echo "============================================================"
echo "  Starting Experimental Runs"
echo "============================================================"
echo ""

for i in {0..2}; do
    subject=${subjects[$i]}
    run=${runs[$i]}

    if run_experiment $subject $run; then
        results[$i]="SUCCESS"
        ((success_count++))
    else
        results[$i]="FAILED"
    fi

    sleep 2  # Brief pause between experiments
done

echo ""
echo "============================================================"
echo "  Experimental Run Summary"
echo "============================================================"
echo ""

for i in {0..2}; do
    subject=${subjects[$i]}
    run=${runs[$i]}
    status=${results[$i]}

    if [ "$status" == "SUCCESS" ]; then
        echo -e "${GREEN}✓ Subject $subject, Run $run: SUCCESS${NC}"
    else
        echo -e "${RED}✗ Subject $subject, Run $run: FAILED${NC}"
    fi
done

echo ""
echo "Success Rate: $success_count/$total_count"
echo ""

# Extract results and update EXPERIMENTAL_RESULTS.md
echo "============================================================"
echo "  Extracting Results"
echo "============================================================"
echo ""

if [ $success_count -eq 0 ]; then
    echo -e "${RED}No experiments succeeded. Cannot generate results summary.${NC}"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi

# Parse log file for accuracy metrics
echo "Parsing experimental results from log file..."
python3 << 'EOF'
import re
import sys
from pathlib import Path

# Read the log file
log_file = sorted(Path("results").glob("experiment_log_*.txt"))[-1]
with open(log_file, 'r') as f:
    log_content = f.read()

# Regular expressions to extract metrics
cv_mean_pattern = r"Cross-validation mean: ([\d.]+)"
cv_std_pattern = r"\+/- ([\d.]+)"
test_acc_pattern = r"Test set accuracy: ([\d.]+)"
pred_time_pattern = r"Average prediction time: ([\d.]+)s"
max_pred_time_pattern = r"Maximum prediction time: ([\d.]+)s"

# Extract all matches
cv_means = re.findall(cv_mean_pattern, log_content)
cv_stds = re.findall(cv_std_pattern, log_content)
test_accs = re.findall(test_acc_pattern, log_content)
pred_times = re.findall(pred_time_pattern, log_content)
max_pred_times = re.findall(max_pred_time_pattern, log_content)

# Print summary
print("Extracted Results:")
print("=" * 60)

subjects = [(1, 6), (4, 14), (10, 6)]
for i, (subj, run) in enumerate(subjects):
    if i < len(cv_means):
        print(f"\nSubject {subj}, Run {run}:")
        print(f"  CV Mean Accuracy: {float(cv_means[i])*100:.2f}%")
        if i < len(cv_stds):
            print(f"  CV Std Dev: ±{float(cv_stds[i])*100:.2f}%")
        if i < len(test_accs):
            print(f"  Test Accuracy: {float(test_accs[i])*100:.2f}%")
        if i < len(pred_times):
            print(f"  Avg Prediction Time: {pred_times[i]}s")
        if i < len(max_pred_times):
            print(f"  Max Prediction Time: {max_pred_times[i]}s")

# Calculate overall statistics
if cv_means:
    cv_mean_values = [float(x) for x in cv_means]
    overall_mean = sum(cv_mean_values) / len(cv_mean_values)
    print(f"\n{'=' * 60}")
    print(f"OVERALL MEAN CV ACCURACY: {overall_mean*100:.2f}%")

    # Check if target is met
    target_threshold = 0.60
    meets_target = all(acc >= target_threshold for acc in cv_mean_values)
    num_meeting_target = sum(1 for acc in cv_mean_values if acc >= target_threshold)

    print(f"Subjects meeting ≥60% target: {num_meeting_target}/{len(cv_mean_values)}")
    print(f"Target Achieved: {'YES ✓' if meets_target else 'NO ✗'}")
    print("=" * 60)

    sys.exit(0 if meets_target else 1)
else:
    print("No results found in log file.")
    sys.exit(1)
EOF

parse_exit_code=$?

echo ""

# Final message
echo "============================================================"
echo "  Experiments Complete"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - Log file: $LOG_FILE"
echo "  - Plots: plots/"
echo "  - Models: models/"
echo ""
echo "Next steps:"
echo "  1. Review EXPERIMENTAL_RESULTS.md (manually update with exact values)"
echo "  2. Examine plots in plots/ directory"
echo "  3. Include results in project defense/presentation"
echo ""

if [ $parse_exit_code -eq 0 ]; then
    echo -e "${GREEN}SUCCESS: System achieves ≥60% accuracy threshold!${NC}"
    exit 0
else
    echo -e "${YELLOW}WARNING: Some subjects may not meet the 60% threshold.${NC}"
    echo "Review the log file for detailed results."
    exit 1
fi
