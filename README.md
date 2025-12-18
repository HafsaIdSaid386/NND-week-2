# NND-week-2
Code Summary 

This code creates a browser-based web application for predicting Titanic survival using a shallow neural network built with TensorFlow.js. The interface allows users to upload the Titanic training and test CSV files, preview the data, and preprocess features such as passenger class, sex, age, family size, and whether a passenger is alone.

The app builds a shallow neural network with one hidden layer (ReLU activation) and a sigmoid output layer for binary classification. Users can train the model by selecting the number of epochs and batch size, then evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix with an adjustable classification threshold.

The application also supports predicting survival on the test dataset, exporting predictions, and saving the trained model. All computation runs entirely in the browser, and the app is designed to be deployed easily as a single-page site using GitHub Pages.
