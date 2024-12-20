# IntruderDetectionForSurveillanceCameras
This application uses the convolution neural networks along with ResNet-50 deep learning model, and datasets to enable facial detection in cameras.

The python application can be used to detect if there is an intruder or just a civilian. The approach uses a dataset of 13.5k images which includes individuals wearing facial coverings as well as regular headshots to train the CNN's model into detecting intruder with mask. 

This was built with the problem that regular security camera's are often times useless when robbers are wearing masks. This technology can identify these suspects and allert the owner immediately to take action.
For winter times, there is a 5 second wait to allow any non-intruders to clearly show their face to the camera and not allert the owner.

# In order to install it and open it on an IDE follow the instructions:
The project that we have created is a Python project, so in order to run it properly here are some aspects to consider:

Python Version:
For our code the version of Python has to be from 3.7 to 3.9 in order for this project to work properly, in order to check which version of Python you have, write the following command in a terminal of your preference:
Python –-version

It should give you the following

This will tell you which version of Python you have, incase you have the wrong version of Python or you don’t have Python at all, here is the link to download Python:
https://www.python.org/downloads/release/python-390/
Just ensure to add Python 3.9.0 to your PATH while downloading.

# Install Python Libraries and TenserFlow Version:
Next install Python libraries needed to run this code, these libraries include tensorflow, imutils, opencv-python, win10toast and numpy, :
pip install tensorflow imutils opencv-python win10toast numpy
The TenserFlow has to be 2.9.0 in order for this project to work, in order to check the version of TensorFlow you have create a .py file, and add the following code:
import tensorflow as tf


print(tf. __version__)
It should give you the following:

If you have the incorrect version, below is a possible solution to the issue, this method should fix the version of TensorFlow but it is not always the solution depending on the issue that you are facing:
pip uninstall tensorflow numpy


pip install tensorflow==2.9 numpy==1.21


pip install --upgrade typing-extensions protobuf
Running the Project:

Now using any IDE that you like (we are using VS Code for our project) open the project folder, and access it through a terminal, once that is done, run the following command:
python DetectIntruder.py
