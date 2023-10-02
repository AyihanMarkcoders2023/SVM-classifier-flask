import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template
import werkzeug
import os

# Load the training and testing data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# Extract the description and category columns
X_train = train_df['DESCRIPTION']
y_train = train_df['CATEGORY']
X_test = test_df['DESCRIPTION']
y_test = test_df['CATEGORY']
# Encode labels into numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Create and train a linear SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train_encoded)

app = Flask(__name__)

# Load the training data
train_df = pd.read_csv('train.csv')

# Extract the description and category columns
X_train = train_df['DESCRIPTION']
y_train = train_df['CATEGORY']

# Encode labels into numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Create and train a linear SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train_encoded)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file temporarily
            uploaded_file.save(uploaded_file.filename)

            # Load the uploaded file for prediction
            predict_df = pd.read_excel(uploaded_file.filename)

            X_predict_tfidf = tfidf_vectorizer.transform(predict_df['DESCRIPTION'])

            # Predict numerical labels for the descriptions
            predicted_labels_encoded = svm_classifier.predict(X_predict_tfidf)

            # Decode the numerical labels back to string categories
            predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

            # Add the predicted categories to the DataFrame
            predict_df['CATEGORY'] = predicted_labels

            # Save the updated DataFrame with predictions to a new Excel file
            prediction_filename = 'testing_' + uploaded_file.filename
            predict_df.to_excel(prediction_filename, index=False, engine='openpyxl')

            # Remove the temporary uploaded file
            os.remove(uploaded_file.filename)

            return "Predictions saved to {}".format(prediction_filename)

    return render_template('testing.html')

if __name__ == '__main__':
    app.run(debug=True)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)
# Calculate accuracy and classification report
accuracy = accuracy_score(y_test_encoded, y_pred)
classification_rep = classification_report(y_test_encoded, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(classification_rep)
# Load the Predict Set v1.xlsx file
predict_df = pd.read_excel('Predict_set_v1.xlsx')

X_predict_tfidf = tfidf_vectorizer.transform(predict_df['DESCRIPTION'])

# Predict numerical labels for the descriptions
predicted_labels_encoded = svm_classifier.predict(X_predict_tfidf)

# Decode the numerical labels back to string categories
predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

# Add the predicted categories to the DataFrame
predict_df['CATEGORY'] = predicted_labels

# Save the updated DataFrame with predictions to a new Excel file
predict_df.to_excel('Predict_set_v1_predictions.xlsx', index=False, engine='openpyxl')
