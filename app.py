from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="template")

classifier = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def hello_world():
     return render_template("index.html")


@app.route("/predict", methods=["POST"])
def home():
   
    d2 = float(request.form["b"])
    d3 = float(request.form["c"])
    d4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 =float(request.form["j"])
    d11 =float(request.form["k"])
    d12 =float(request.form["l"])
    
   
   

    arr = np.array(
        [
            [
                
                d2,
                d3,
                d4,
                d5,d6,d7,d8,d9,d10,d11,d12
                            ]
        ]
    )
    pred = classifier.predict(arr)
    print(pred)
    return render_template("result.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)