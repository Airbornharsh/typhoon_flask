import json
import os
from flask import Flask, jsonify, request
from utils import fileOperations
from model import student_thai

typhoon_model, typhoon_tokenizer = None, None
custom_model_and_tokenizer = {"thai_typhoon_model": {"data": (None, None), "index": 0}}

app = Flask(__name__)

# if typhoon_model is None or typhoon_tokenizer is None:
#     typhoon_model, typhoon_tokenizer = student_thai.load_model()
#     print("Loaded typhoon model")

@app.route("/", methods=["GET"])
def get_employees():
    return jsonify({"message": "Hello, World!"})


@app.route("/upload/<string:id>", methods=["POST"])
def upload_file(id):
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]

    folder_path = "./sheets/" + id + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    id_items = os.listdir(folder_path)
    new_id = len(id_items) + 1

    folder_path += str(new_id) + "/"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if file.filename == "":
        return "No selected file"
    file.save("./sheets/" + id + "/" + str(new_id) + "/data.csv")
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
    try:
        folder_path = "./sheets/" + id + "/"
        if not os.path.exists(folder_path):
            return "No files found"
        id_items = os.listdir(folder_path)

        global typhoon_model, typhoon_tokenizer
        if typhoon_model is None or typhoon_tokenizer is None:
            typhoon_model, typhoon_tokenizer = student_thai.load_model()
        print("Loaded typhoon model")
        index = fileOperations.highest_numbered_file(id_items)

        if id not in custom_model_and_tokenizer:
            custom_model_and_tokenizer[id] = {
                "data": (None, None),
                "index": 0,
            }

        print("Intialized custom model and tokenizer for ", id)
        print(index)
        print(custom_model_and_tokenizer)

        if index != custom_model_and_tokenizer[id]["index"]:
            print("Index Found")
            temp_model, temp_tokenizer = student_thai.load_data_and_train(
                typhoon_model,
                typhoon_tokenizer,
                id,
            )
            print("Loaded Data and Train")
            custom_model_and_tokenizer[id] = {
                "data": (temp_model, temp_tokenizer),
                "index": index,
            }
            print("Loaded Data and Train 2")
        # else:
        #     print("No index Found")
        #     temp_model, temp_tokenizer = student_thai.load_data_and_train(
        #         custom_model_and_tokenizer[id]["data"][0],
        #         custom_model_and_tokenizer[id]["data"][1],
        #         id,
        #     )
        #     print("Loaded Data and Train")
        #     custom_model_and_tokenizer[id] = {
        #         "data": (temp_model, temp_tokenizer),
        #         "index": index,
        #     }
        #     print("Loaded Data and Train 2")

        return json.dumps("Model training has started")
    except Exception as e:
        return json.dumps(str(e))


@app.route("/predict/<string:id>", methods=["POST"])
def model_predict(id):
    print("Model loaded for ", id)
    return json.dumps("Model prediction has started")


if __name__ == "__main__":
    app.run(port=8000, debug=True)
