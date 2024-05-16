# Deep-Learning

## Install TensorFlow on MacOS using Homebrew

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

source tf-env/bin/activate


### Install TensorFlow

pip install tensorflow

Or if requiring GPU support on Mac M1, using

pip install tensorflow-macos

pip install tensorflow-meta1


### Verify the installation:

python3 -c "import tensorflow as tf; print(tf.__version__)"


## MIT Introduction to Deep Learning | 6.S191

### Download and import the course package

pip install mitdeeplearning --quiet

### Download and import CV2 module

pip install opencv-python


## Update PyComplexHeatmap module

### Manually remove its old version

rm -rf /opt/homebrew/lib/python3.12/site-packages/PyComplexHeatmap*

### Install the latest version

python3 -m pip install PyComplexHeatmap

### Verify the installation

pip show PyComplexHeatmap   

or 

pip3 show PyComplexHeatmap   
