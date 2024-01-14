from flask import Flask, request, jsonify, render_template
from md import answer
app = Flask(__name__)
 
 
@app.route("/")
def index():

    return render_template("index.html", failure=None)
 
@app.route('/data', methods=["POST", "GET"])
def login():
    inps = [float(i) for i in request.form.values()]
    failure = answer(inps)
    return render_template("index.html", failure=failure)

if __name__ == '__main__':
    app.run(debug=True, port=8123)
