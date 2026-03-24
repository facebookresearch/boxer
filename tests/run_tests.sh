#!/bin/bash
# Run tests with coverage and generate an HTML report.
#
# Usage:
#   ./run_tests.sh                    # run all tests
#   ./run_tests.sh test_gravity       # run a single test file
#   ./run_tests.sh test_gravity.py    # also works with .py extension
#   ./run_tests.sh --no-open          # run without opening the report
#
# Requires: pip install pytest pytest-cov

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Ensure test dependencies are installed
python -m pip install -q pytest pytest-cov

# Parse arguments: first non-flag arg is the test selector
TEST_TARGET="tests/"
NO_OPEN=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-open" ]]; then
        NO_OPEN=true
    elif [[ "$arg" != -* && "$TEST_TARGET" == "tests/" ]]; then
        name="${arg%.py}"
        TEST_TARGET="tests/${name}.py"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# Clear previous coverage data
python -m coverage erase

# Run pytest with coverage data collection only (no HTML yet)
python -m pytest "$TEST_TARGET" \
    --cov=boxernet --cov=utils --cov=detectors --cov=loaders \
    --cov-config=tests/.coveragerc \
    --cov-report=term \
    -v "${EXTRA_ARGS[@]}"

# Build the HTML report, omitting files with 0% coverage
python -c "
import coverage, os, shutil

cov = coverage.Coverage(config_file='tests/.coveragerc')
cov.load()

# Find files that were actually executed
touched = []
for f in cov.get_data().measured_files():
    # Only include project files
    rel = os.path.relpath(f)
    if rel.startswith(('boxernet/', 'utils/', 'detectors/', 'loaders/')):
        analysis = cov._analyze(f)
        if len(analysis.executed) > 0:
            touched.append(rel)

# Write a temporary .coveragerc that omits everything except touched files
outdir = 'tests/htmlcov'
if os.path.isdir(outdir):
    shutil.rmtree(outdir)

cov2 = coverage.Coverage(config_file='tests/.coveragerc', include=touched)
cov2.load()
cov2.html_report(directory=outdir)
"

echo ""
echo "==> HTML coverage report: tests/htmlcov/index.html"

if [[ "$NO_OPEN" == false ]]; then
    open tests/htmlcov/index.html 2>/dev/null || true
fi
