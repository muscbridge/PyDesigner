#!/bin/sh

# If no arguments are passed, run pytest with coverage
if [ "$#" -eq 0 ]; then
    pytest tests -vv  --cov=. --cov-report=xml:/tests/coverage.xml --junitxml=/tests/results.xml
else
    # Otherwise, run the command passed as arguments
    exec "$@"
fi
