#!/opt/homebrew/bin/python3
import os
import subprocess
import argparse


def main():
    # Default pipeline path values
    PIPELINE_DIR = "src"
    PIPELINE_FILE = "pipeline.py"

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Basketball analysis pipeline')
    parser.add_argument('--p', dest='pipeline_path', help='Path to the pipeline file to run on Modal')
    # parser.add_argument('--q', type=str, help='Queries to run, separated by ";", surrounded by double quotes')

    # Parse known args to extract the pipeline string if present
    args, remaining_args = parser.parse_known_args()

    # If pipeline string was provided, extract it and pass the remaining args to the pipeline
    pipeline_path = args.pipeline_path
    if pipeline_path:
        # Get the index of the last "/" in the pipeline path
        last_slash_index = pipeline_path.rfind("/")

        PIPELINE_DIR = pipeline_path[:last_slash_index]
        PIPELINE_FILE = pipeline_path[last_slash_index + 1:]

    print(f"Running pipeline from {PIPELINE_DIR}/{PIPELINE_FILE}")
    # Convert pipeline_args list to a space-separated string, with special handling with
    #   any string with spaces (which is presumably an argument's value)
    pipeline_args_str = ""
    for remaining_arg in remaining_args:
        if " " in remaining_arg:
            pipeline_args_str += f'"{remaining_arg}" '
        else:
            pipeline_args_str += f"{remaining_arg} "
    print(f"Arguments being passed to the pipeline: {pipeline_args_str}")

    # Get path to env setup script
    current_dir = os.getcwd()
    setup_script_path = "utils/setup.sh"

    # Setup environment and run modal in the same shell process
    print("Sourcing setup script then running pipeline...")

    # Run both commands in a single shell process
    full_cmd = (
        f"source {setup_script_path} {PIPELINE_DIR}/requirements.txt && "
        f"modal run {PIPELINE_DIR}/{PIPELINE_FILE} {pipeline_args_str}"
    )
    print(f"Full command: {full_cmd}")
    while True:
        result = subprocess.run(full_cmd, shell=True, executable="/bin/bash")
        exit_code = result.returncode

        # Retry if we got the exit code 20: this is our exit code
        #   for a "video not available in your country" yt-dlp error,
        #   which requires a restart.
        # Otherwise, end the program (pipeline either completed
        #   or encountered some other error).
        if exit_code != 20:
            break


if __name__ == "__main__":
    main()
