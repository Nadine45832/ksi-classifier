from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("ksi_classifier.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/", methods=["GET"])
def form():
    return render_template("form.html", prediction=None, probability=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "LATITUDE": float(request.form["LATITUDE"]),
            "LONGITUDE": float(request.form["LONGITUDE"]),
            "ACCLOC": request.form["ACCLOC"],
            "TRAFFCTL": request.form["TRAFFCTL"],
            "LIGHT": request.form["LIGHT"],
            "IMPACTYPE": request.form["IMPACTYPE"],
            "INVTYPE": request.form["INVTYPE"],
            "INVAGE": int(request.form["INVAGE"]),
            "INJURY": request.form["INJURY"],
            "DRIVACT": request.form["DRIVACT"],
            "DRIVCOND": request.form["DRIVCOND"],
            "PEDTYPE": request.form["PEDTYPE"],
            "PEDESTRIAN": request.form["PEDESTRIAN"],
            "CYCLIST": request.form["CYCLIST"],
            "AUTOMOBILE": request.form["AUTOMOBILE"],
            "SPEEDING": request.form["SPEEDING"],
            "AG_DRIV": request.form["AG_DRIV"],
        }

        df = pd.DataFrame([data])
        transformed = pipeline.transform(df)
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0][1]

        return render_template("form.html", prediction=prediction, probability=round(probability, 4))

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
