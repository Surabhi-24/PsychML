from flask import Flask, app, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None
    text=""
    if request.method == "POST":
        text = request.form["user_text"]
        # Here you would typically process the text and predict emotion
        emotion = "Predication will appear here"
    return render_template("index.html", emotion=emotion, text=text)

if __name__ == "__main__":
    app.run(debug=True)
    