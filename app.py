from flask import Flask, render_template, request
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the Support Vector Machine (SVM) model
model = SVC(kernel='linear', random_state=42)

# Train the model
model.fit(X, y)

# Save the model (if not done already)
joblib.dump(model, 'iris_svm_model.pkl')

# Initialize the scaler (to standardize the input data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Define a function to preprocess input data and predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Input data preprocessing
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    data = scaler.transform(data)  # Standardize the input data

    # Make prediction
    prediction = model.predict(data)
    return iris.target_names[prediction][0]

# Route for the homepage (HTML form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Get the prediction from the model
        predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        
        return render_template('index.html', prediction_text=f"Predicted Iris Species: {predicted_species}")

if __name__ == '__main__':
    app.run(debug=True)
