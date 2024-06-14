import pandas as pd  # Import pandas for data manipulation and analysis
import os  # Import os for interacting with the operating system (e.g., reading directories)
import re  # Import re for regular expressions (e.g., text preprocessing)
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for text vectorization
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data into train and test sets
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression for the classification model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Import metrics for model evaluation

# Get the directory of the current script to build absolute paths for data directories
base_dir = os.path.dirname(os.path.abspath(__file__))

def read_spam():
    """Read spam emails from the 'spam' directory"""
    category = 'spam'  # Define the category as spam
    directory = os.path.join(base_dir, 'data', 'spam')  # Construct the absolute path to the spam directory
    return read_category(category, directory)  # Read emails from the spam directory

def read_ham():
    """Read ham (non-spam) emails from the 'ham' directory"""
    category = 'ham'  # Define the category as ham
    directory = os.path.join(base_dir, 'data', 'ham')  # Construct the absolute path to the ham directory
    return read_category(category, directory)  # Read emails from the ham directory

def read_category(category, directory):
    """Read all emails from a given directory and categorize them"""
    print(f"Reading category: {category}, from directory: {directory}")  # Print the current operation
    emails = []  # Initialize an empty list to store email data
    for filename in os.listdir(directory):  # Iterate over all files in the directory
        if not filename.endswith(".txt"):  # Skip files that are not .txt
            continue  # Move to the next file
        with open(os.path.join(directory, filename), 'r') as fp:  # Open the file
            try:
                content = fp.read()  # Read the content of the file
                emails.append({'name': filename, 'content': content, 'category': category})  # Append email details to the list
            except Exception as e:
                print(f'skipped {filename}: {e}')  # Print an error message if reading the file fails
    return emails  # Return the list of emails

# Print current working directory and data directories for debugging purposes
print(f"Current working directory: {os.getcwd()}")
print(f"Ham directory: {os.path.abspath(os.path.join(base_dir, 'data', 'ham'))}")
print(f"Spam directory: {os.path.abspath(os.path.join(base_dir, 'data', 'spam'))}")

# Read emails from both categories
ham = read_ham()  # Read ham emails
spam = read_spam()  # Read spam emails

# Create DataFrames from the read emails
df_ham = pd.DataFrame.from_records(ham)  # Convert the ham emails list to a DataFrame
df_spam = pd.DataFrame.from_records(spam)  # Convert the spam emails list to a DataFrame

# Concatenate the DataFrames into a single DataFrame
df = pd.concat([df_ham, df_spam], ignore_index=True)  # Combine the ham and spam DataFrames

def preprocessor(e):
    """Preprocess email content by removing non-alphabetic characters and converting to lowercase"""
    return re.sub('[^A-Za-z]', ' ', e).lower()  # Replace non-alphabetic characters with spaces and convert to lowercase

# Instantiate a CountVectorizer with the custom preprocessor
vectorizer = CountVectorizer(preprocessor=preprocessor)

# Split the dataset into training and testing sets
# X contains email content and y contains the email category (ham or spam)
X_train, X_test, y_train, y_test = train_test_split(df["content"], df["category"], test_size=0.2, random_state=1)

# Fit the vectorizer on the training data and transform it into a document-term matrix
X_train_df = vectorizer.fit_transform(X_train)

# Instantiate and train the logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train_df, y_train)

# Transform the test data using the fitted vectorizer
X_test_df = vectorizer.transform(X_test)

# Predict the categories of the test data
y_pred = model.predict(X_test_df)

# Print the evaluation metrics
# `accuracy_score` provides the fraction of correctly classified samples
# `confusion_matrix` shows the number of correct and incorrect predictions for each class
# `classification_report` provides a detailed report including precision, recall, and F1-score for each class
print(f'Accuracy:\n{accuracy_score(y_test, y_pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
print(f'Detailed Statistics:\n{classification_report(y_test, y_pred)}\n')

# Get the feature names (words) from the vectorizer
features = vectorizer.get_feature_names_out()

# Get the importance (coefficients) of the features from the logistic regression model
# The coefficients indicate the importance of each feature (word) in predicting the target variable
importance = model.coef_[0]

# Create a list of tuples with feature index and importance
l = list(enumerate(importance))

# Print the top 10 most important features for predicting spam
print("\nTop 10 positive features (indicating spam):")
l.sort(key=lambda e: e[1], reverse=True)  # Sort features by importance in descending order
for i, imp in l[:10]:  # Loop through the top 10 features
    print(f"{imp:.4f}", features[i])  # Print the importance and feature name

# Print the top 10 least important features for predicting spam (indicating ham)
print("\nTop 10 negative features (indicating ham):")
l.sort(key=lambda e: -e[1], reverse=True)  # Sort features by importance in ascending order
for i, imp in l[:10]:  # Loop through the top 10 features
    print(f"{imp:.4f}", features[i])  # Print the importance and feature name
