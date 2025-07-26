from flask import Flask, render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method =="POST":
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        return render_template("index.html",prediction_text=f"the news is: {result}")
    
if __name__ == "__main__":
    app.run(debug=True)
    