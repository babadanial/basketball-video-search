#!/bin/bash
# Arguments: 
# - $1 = requirements file path

# This script:
#  - installs Python 3.10 locally
#  - creates and activates the virtual environment
#  - installs the dependencies for the app from the requirements file (specified using first arg) 
#    into the virtual environment
REQUIREMENTS_FILE=${1:-"src/requirements.txt"}
PYTHON_VERSION="3.10"
VENV_DIR=".venv-$PYTHON_VERSION"

echo "================================================================================================"
echo " 🔨⚙️  Running pipeline environment setup script..."

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew update
  brew upgrade
  brew install -q python@"$PYTHON_VERSION"
else
  sudo apt-get -qq update
  sudo apt-get -qq install -y python"$PYTHON_VERSION"
fi

# delete the venv if it is not the Python version we want
if [[ -d "$VENV_DIR" ]]; then
  if [[ ! -f "$VENV_DIR"/bin/python"$PYTHON_VERSION" ]]; then
    echo "🐍 $VENV_DIR/bin/python$PYTHON_VERSION does not exist; thus either:"
    echo "🐍 - wrong Python version is installed in the $VENV_DIR directory"
    echo "🐍 - $VENV_DIR is not a Python venv"
    echo "🐍 Deleting the existing Python venv directory at $VENV_DIR..."
    rm -rf $VENV_DIR
  fi
fi

mkdir -p $VENV_DIR
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "🐍 $VENV_DIR is not a Python venv. Creating a new Python venv..."
  python"$PYTHON_VERSION" -m venv $VENV_DIR
fi

echo "🐍 Activating the Python $PYTHON_VERSION venv at $VENV_DIR..."
source "$VENV_DIR"/bin/activate

echo "🐍 Upgrading pip and installing packages"
pip install -qq --upgrade pip
pip install -qq -r $REQUIREMENTS_FILE

echo " 🔨⚙️  Pipeline environment setup script complete."
echo "================================================================================================"