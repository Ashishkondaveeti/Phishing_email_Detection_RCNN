from flask import Flask, render_template, request
from rcnn import load_model_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['emailContent']
    phishing_probability = load_model_and_predict(email_content)
    return render_template('index.html', prediction=True, phishing_probability=phishing_probability)

if __name__ == '__main__':
    app.run(debug=True)
