#!/bin/bash

# If no arguments are passed, run pytest with coverage
if [ "$#" -eq 0 ]; then
    mkdir -p /test_results
    pytest tests -vv  --cov=pydesigner --cov-report=xml:/test_results/coverage.xml --junitxml=/test_results/results.xml
else
    # Otherwise, run the command passed as arguments
    exec "$@"
fi
