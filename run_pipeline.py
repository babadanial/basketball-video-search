#!/opt/homebrew/bin/python3
import os
import sys
import subprocess
import argparse


def main():
    # Default pipeline path values
    PIPELINE_DIR = "src"
    PIPELINE_FILE = "pipeline.py"

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Basketball analysis pipeline')
    parser.add_argument('--p', dest='pipeline_path', help='Path to the pipeline file to run on Modal')

    # Parse known args to extract the pipeline string if present
    args, remaining_args = parser.parse_known_args()

    # If pipeline string was provided, extract it and pass the remaining args to the pipeline
    pipeline_path = args.pipeline_path
    if pipeline_path:
        # Get the index of the last "/" in the pipeline path
        last_slash_index = pipeline_path.rfind("/")

        PIPELINE_DIR = pipeline_path[:last_slash_index]
        PIPELINE_FILE = pipeline_path[last_slash_index + 1:]

        if "--p" in sys.argv:
            p_index = sys.argv.index("--p")
            # Remove --p and its value
            sys.argv.pop(p_index)  # Remove --p
            if p_index < len(sys.argv):
                sys.argv.pop(p_index)  # Remove the value of --p

    print(f"Running pipeline from {PIPELINE_DIR}/{PIPELINE_FILE}")
    pipeline_args = sys.argv[1:]
    # Convert pipeline_args list to a space-separated string
    pipeline_args_str = ' '.join(pipeline_args)
    print(f"Arguments being passed to the pipeline: {pipeline_args_str}")

    # Get path to env setup script
    current_dir = os.getcwd()
    setup_script_path = os.path.join(current_dir, "utils/setup.sh")

    # Go to the script directory
    os.chdir(os.path.join(PIPELINE_DIR))
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Changed working directory to: {os.getcwd()}")

    # Setup environment and run modal in the same shell process
    print("Sourcing setup script then running pipeline...")

    # Run both commands in a single shell process
    full_cmd = f"source {setup_script_path} && modal run {PIPELINE_FILE} {pipeline_args_str}"
    while True:
        result = subprocess.run(full_cmd, shell=True, executable="/bin/bash")
        exit_code = result.returncode

        # Retry if we got the exit code 20: this is our exit code
        #   for a "video not available in your country" error, which
        #   requires a restart.
        # Otherwise, end the program (pipeline either completed
        #   or encountered some other error).
        if exit_code != 20:
            break


if __name__ == "__main__":
    main()
