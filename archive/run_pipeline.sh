#!/bin/bash
# Must be run from the root of the project directory
# Usage: ./run_pipeline.sh [--p path/to/pipeline/file] [pipeline args]

PIPELINE_DIR="embedding-based"
PIPELINE_FILE="pipeline.py"
# Parse arguments to extract --p option
PIPELINE_ARGS=()
PIPELINE_OPTION=""

# Save current directory to return to it after execution
PROJECT_ROOT_DIR=$(pwd)

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--p" ]]; then
    if [[ $# -gt 1 ]]; then
      PIPELINE_OPTION="$2"
      # Find the last occurrence of "/" to determine the pipeline directory
      if [[ "$PIPELINE_OPTION" == */* ]]; then
        PIPELINE_DIR="${PIPELINE_OPTION%/*}"  # Remove everything after the last "/"
        PIPELINE_FILE="${PIPELINE_OPTION##*/}" # Get everything after the last "/"
        echo "Using pipeline: $PIPELINE_DIR/$PIPELINE_FILE"
      fi
      shift 2
    else
      echo "Error: --p option requires a value"
      exit 1
    fi
  else
    PIPELINE_ARGS+=("$1")
    shift
  fi
done

# Reset positional parameters to the filtered arguments
set -- "${PIPELINE_ARGS[@]}"

cd "$PIPELINE_DIR" || (echo "Error: Pipeline directory not found" && exit 1)
source "$PROJECT_ROOT_DIR"/utils/setup.sh;
modal run "$PIPELINE_DIR"/"$PIPELINE_FILE" "$@";