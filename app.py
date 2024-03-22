import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import nltk
nltk.download('all')
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the TF-IDF vectorizer and the trained Decision Tree model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('decision_tree.pkl')

# Define the preprocess_text function
def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in text_data:
        # Removing punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower()
                                          for token in nltk.word_tokenize(sentence)
                                          if token.lower() not in stopwords.words('english')))

    return preprocessed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input from the form
        review_text = request.form.get('review')

        # Preprocess the input text
        preprocessed_text = preprocess_text([review_text])

        # Transform the preprocessed text using the TF-IDF vectorizer
        text_vector = vectorizer.transform(preprocessed_text).toarray()

        # Make a prediction using the model
        prediction = model.predict(text_vector)

        # Determine the sentiment label
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', prediction=sentiment)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
