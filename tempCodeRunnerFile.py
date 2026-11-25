
from run_predict import main as run_prediction
from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        result = run_prediction()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
