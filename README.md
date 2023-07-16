# ann
- neural_network.py:
  - self implementation of a neural network that can be initialized with a variable number of layers and nodes per layer
- main.py:
  -  training and testing of my own implementation of the neural network over iris dataset and handwritten digits dataset (sklearn)
  -  I've tried using various amounts of data (from the available set) for training and testing, mostly circling around 70% - 75% - 80% (I've also, absurdely, used lower percentages just for the sake of curiosity).
  -  I've tried using early stop mechanisms to prevent overfitting and decreasing the learning rate depending on the standard deviation of the error over the last few iterations.
  -  Achieved above 90% accuracy on most tests.

- main_tool.py -> training and testing of a tensorflow MLP and CNN over 600x600 input images with or without a sepia filter applied to them (filter.py for more details)
