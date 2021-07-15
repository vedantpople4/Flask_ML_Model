from flask import Flask, render_template, request
import marks as m
app  = Flask(__name__)


@app.route("/", methods=['GET','POST'])
def marks():
    if request.method == "POST":
        hrs = request.form['hrs']
        marks_pred = m.marks_prediction(hrs)
        # mks = marks_pred
    return render_template("submit.html", n = marks_pred)


"""
@app.route("/submit", methods=['POST'])
def submit():
    if request.method == "POST":
        name = request.form["username"]
    
    return render_template("submit.html", n = name )
"""
if __name__ == "__main__":
    app.run(debug=True)
