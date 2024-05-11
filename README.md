# Deep-Learning

MIT Introduction to Deep Learning | 6.S191

## Download and import the MIT Introduction to Deep Learning package

pip3 install mitdeeplearning --quiet


## Install tensorflow on MacOS using Homebrew

### Update Homebrew

brew update

### Install or update python using Homebrew

brew install python or brew upgrade python

### Check the veresions of python3 and pip3

python3 --version

pip3 --version

### Set up a vertual environment using venv

python3 -m venv tf-env

### Activate the virtual environment named "tf-env"

surce tf-env/bin/activate

### Install TensorFlow

pip install tensorflow

Or if requiring GPU support on Mac M1, using

pip install tensorflow-macos

pip install tensorflow-meta1


## Verify the installation:

python3 -c "import tensorflow as tf; print(tf.__version__)"
