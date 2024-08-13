from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

# Load the model
with open('/users/nik/digit_recognition_app/digit_recognition_cnn.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the POST request
    img = Image.open(request.files['digit'])
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors for better recognition
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Predict the digit
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    
    return render_template('index.html', prediction_text=f'Predicted Digit: {predicted_digit}')

if __name__ == '__main__':
    app.run(debug=True)
