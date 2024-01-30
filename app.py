from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Create flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods = ["POST"])
def predict():
    input=request.json
    features=pd.DataFrame(input)
    prediction = model.predict(features)
    return jsonify({"Prediction":list(prediction)})

if __name__ == "__main__":
    app.run(debug=True)