import json
import os
from flask import Flask, jsonify, request
from utils import fileOperations
from model import student_thai

typhoon_model, typhoon_tokenizer = None, None

app = Flask(__name__)


@app.route("/", methods=["GET"])
def get_employees():
    return jsonify({"message": "Hello, World!"})


@app.route("/upload/<string:id>", methods=["POST"])
def upload_file(id):
    folder_path = "./sheets/" + id + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    id_items = os.listdir(folder_path)
    new_id = len(id_items) + 1

    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    # Here you can save the file to a desired location or process it
    # For example, save it to a temporary directory
    file.save("./sheets/" + id + "/" + str(new_id) + ".csv")
    return "File uploaded successfully"


@app.route("/file/<string:id>", methods=["GET"])
def get_files(id):
    folder_path = "./sheets/" + id + "/"
    if not os.path.exists(folder_path):
        return "No files found"
    id_items = os.listdir(folder_path)
    file_name = fileOperations.highest_numbered_file(id_items)
    return json.dumps(file_name)


@app.route("/train/<string:id>", methods=["GET"])
def model_train(id):
    global typhoon_model, typhoon_tokenizer
    if typhoon_model is None or typhoon_tokenizer is None:
        typhoon_model, typhoon_tokenizer = student_thai.load_model()
    print("Model loaded for ", id)
    return json.dumps("Model training has started")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
