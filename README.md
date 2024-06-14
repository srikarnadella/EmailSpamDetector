# Email Spam Detector

## Overview

This project is an Email Spam Detector using a Logistic Regression model. It processes email text data, categorizes it into spam and ham (non-spam), and trains a machine learning model to classify new emails. The model uses a `CountVectorizer` to convert text data into numerical features and a Logistic Regression model to make predictions.

## Directory Structure

EmailSpamDetector/
│
├──data/
│ ├──ham/
│ │ └──[ham_email_1.txt, ham_email_2.txt, ...]
│ └──spam/
│ └──[spam_email_1.txt, spam_email_2.txt, ...]
│
└──SpamDetection.py


- `data/ham/`: Directory containing ham (non-spam) emails as `.txt` files.
- `data/spam/`: Directory containing spam emails as `.txt` files.
- `SpamDetection.py`: The main script that reads the emails, preprocesses the data, trains the model, and evaluates it.

## Requirements

- Python 3.x
- Pandas
- scikit-learn

## Explanation

### Main Components

1. **Reading Emails**:
    - The `read_category` function reads all `.txt` files from a given directory and categorizes them.
    - `read_ham` and `read_spam` functions call `read_category` for ham and spam directories, respectively.

2. **Preprocessing**:
    - The `preprocessor` function removes non-alphabetic characters and converts text to lowercase.

3. **Vectorization**:
    - The `CountVectorizer` converts text data into a matrix of token counts, using the custom preprocessor.

4. **Model Training**:
    - The dataset is split into training and testing sets using `train_test_split`.
    - The `LogisticRegression` model is trained on the vectorized training data.

5. **Evaluation**:
    - The trained model is tested on the test data.
    - Accuracy, confusion matrix, and a detailed classification report are printed.
    - The most important features (words) for predicting spam and ham are displayed.

### Detailed Code Comments

The code includes detailed comments to explain each step and the functionality of various components. This makes it easier to understand and modify the code as needed.

### Sample Output

'
Accuracy:
0.95

Confusion Matrix:
[[25  2]
 [ 1 22]]



    Detailed Statistics:
             Precision    Recal    F1-Score    Support
         ham       0.96      0.93      0.95        27
        spam       0.92      0.96      0.94        23

    accuracy                           0.95        50 
    macro avg      0.94      0.95      0.94        50
    weighted avg   0.95      0.95      0.95        50

Top 10 positive features (indicating spam):
1.5872 http
1.2345 free
1.1234 offer


Top 10 negative features (indicating ham):
-1.1234 meeting
-1.0567 project
-0.9876 agenda
'
