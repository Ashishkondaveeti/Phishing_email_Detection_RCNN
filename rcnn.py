from tensorflow.keras.preprocessing.sequence import pad_sequences
from trained_model import load_trained_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load the tokenizer from a JSON file (replace 'tokenizer_path' with the actual path)
tokenizer_path = 'tokenizer.json'
with open(tokenizer_path, 'r') as json_file:
    tokenizer_json = json_file.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Load the trained RCNN model (replace 'model_path' with the actual path)
model_path = 'model.h5'
model = load_trained_model(model_path)

# Set the maximum sequence length used during training
maxlen = 100

# Preprocess function (replace with your actual preprocessing steps)
def preprocess(email_content):
    # Your preprocessing steps go here...
    preprocessed_email = [int(token) for token in email_content.split()]  # Example (replace with actual preprocessed data)
    return preprocessed_email

# Function to make predictions on user input
def load_model_and_predict(email_content):
    preprocessed_email = preprocess(email_content)
    tokenized_email = tokenizer.texts_to_sequences([preprocessed_email])
    padded_email = pad_sequences(tokenized_email, maxlen=maxlen)
    prediction = model.predict(padded_email)[0][0]
    phishing_probability = prediction if prediction >= 0.5 else 1 - prediction
    return phishing_probability
