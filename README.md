# Applicant Data Preprocessing and Binary Classification

## Overview

# This repository contains code for preprocessing applicant data and building a binary classification model using deep neural networks.

## Dataset

# The dataset used for this project is stored in the `applicants_data.csv` file. It contains information about applicants, including various attributes such as application type, affiliation, classification, use case, organization, income amount, special considerations, ask amount, and whether the application was successful or not.

## Instructions

# To run the code in this repository, follow these steps:

# 1. Clone the repository to your local machine.
# 2. Install the required dependencies listed in the `requirements.txt` file by running `pip install -r requirements.txt`.
# 3. Place the `applicants_data.csv` file in the root directory of the repository.

## Imports

# The following libraries are imported in the code:

# import pandas as pd  # for data manipulation and analysis
# import numpy as np  # for numerical computations
# from sklearn.preprocessing import OneHotEncoder  # for encoding categorical variables
# from sklearn.model_selection import train_test_split  # for splitting the data into training and testing sets
# from sklearn.preprocessing import StandardScaler  # for scaling the features data
# from tensorflow.keras.models import Sequential  # for creating the deep neural network model
# from tensorflow.keras.layers import Dense  # for adding dense layers to the model
# from tensorflow.keras.callbacks import ModelCheckpoint  # for saving the model during training
# from tensorflow.keras.models import load_model  # for loading a saved model
# from tensorflow.keras.utils import plot_model  # for visualizing the model architecture
# from tensorflow.keras.models import save_model  # for saving the trained model

## Usage

# The code in this repository is organized into multiple steps. Each step performs a specific task in the preprocessing and binary classification pipeline. Here's an overview of the steps:

# 1. Read and Review the Dataset: The dataset is read into a Pandas DataFrame and reviewed to identify categorical variables and columns defining features and target variables.
# 2. Drop Irrelevant Columns: Irrelevant columns are dropped from the DataFrame.
# 3. Encode Categorical Variables: Categorical variables in the dataset are encoded using the OneHotEncoder from scikit-learn.
# 4. Combine Encoded and Numerical Variables: Encoded categorical variables and original numerical variables are combined into a single DataFrame.
# 5. Create Features and Target Datasets: The features dataset (X) and the target dataset (y) are created.
# 6. Split the Data into Training and Testing Sets: The features and target datasets are split into training and testing datasets.
# 7. Scale the Features Data: The features data is scaled using the StandardScaler from scikit-learn.
# 8. Build the Binary Classification Model: A deep neural network model is built using the Keras API from TensorFlow.
# 9. Compile and Fit the Model: The model is compiled and fitted to the training data.
# 10. Evaluate the Model: The model's performance is evaluated using the testing data.
# 11. Save the Model: The trained model is saved to an HDF5 file.

## Conclusion

# By following the steps in this repository, you can preprocess applicant data and build a binary classification model using deep neural networks. The models can be optimized and fine-tuned to achieve higher predictive accuracy by exploring adjustments to the input data, changing the number of hidden layers and neurons, and experimenting with different activation functions.

## License

# This project is licensed under the MIT License.
