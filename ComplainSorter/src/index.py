import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def Preprocesing():
    return "Hello World"

@app.route('/post', methods=['GET'])
def PostProcessing():
    return "Hello Post Proccesing"

app.run()
