import os
import json
import torch
import shutil
from PIL import Image
import os.path as osp
from fastapi import FastAPI, UploadFile, File
from src.backend.models.foodnet import FoodNet
from torchvision.transforms import ToTensor as tt

# class names
classes = ["Bread","Dairy product","Dessert",
            "Egg","Fried food","Meat","Noodles-Pasta",
            "Rice","Seafood","Soup","Vegetable-Fruit"]

# Update the paths as per your usage
model_conf_path = "src/backend/configs/models/foodnet_base.json"
dataset_conf_path = "src/backend/configs/datasets/food_base.json"
pretrained_weights_path = "pretrained_models/checkpoint1/best_model.pt"


model_conf = json.load(open(model_conf_path, "r"))
dataset_conf = json.load(open(dataset_conf_path, "r"))
MODEL = FoodNet(model_conf, dataset_conf["num_classes"])
MODEL.load_state_dict(torch.load(pretrained_weights_path, map_location="cpu")["model"])
FILENAME = "src/client/responses/temp_image.png"
app = FastAPI()


@app.get("/")
def dashboard():
    return ["This is the Home Page."]

@app.post("/upload")
def upload_image(file: UploadFile = File(...)):
    if file.filename.split(".")[-1] in ["jpg", "png", "jpeg"]:
        with open(FILENAME, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return [200, "File Uploaded to Server."]
    else:
        return [415, "Unsupported Media Format."]


@app.get("/delete")
def delete_image():
    if osp.isfile(FILENAME):
        os.remove(FILENAME)
        return [200, "File Deleted."]
    else:
        return [404, "File Not Found."]

@app.get("/predict")
def predict():
    if osp.isfile(FILENAME):
        image = Image.open(FILENAME)
        image = image.resize((dataset_conf["data_shape"]["h"],
                                dataset_conf["data_shape"]["h"]))
        image = tt()(image).unsqueeze(0)
        prediction = MODEL(image)
        class_id = torch.argmax(torch.softmax(prediction, dim=1),1).detach().numpy()[0]
        class_name = classes[class_id]
        prob = torch.softmax(prediction, dim=1).detach().numpy().reshape(-1).tolist()[class_id]

        return {"class_name": class_name,
                "probability": round(prob*100, 3)}
    else:
        return [404, "File Not Found."]
