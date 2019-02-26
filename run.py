import base64
import json
import os
import re
from io import BytesIO

import numpy as np
from flask import (Flask, flash, jsonify, logging, redirect, render_template,
                   request, session, url_for)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms

app = Flask(__name__, static_folder='static')
model = None
use_gpu = False


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 784)
        self.fc2 = nn.Linear(784, 196)
        self.fc3 = nn.Linear(196, 4)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def load_model():
    """
        Replace the checkpoint saveCheck2.pt with your file.You can obtain the checkpoint
        by using the notebook file.
        
    """
    global model
    
    # Defining the CNN architecture
    model = NeuralNet()

    # Loading the checkpoint
    model.load_state_dict(torch.load('saveCheck2.pt', map_location='cpu'))
    # print(model)


def process_image(image):
    """
    Parameters:
    image : DATA URI

    Returns:

    image_tensor : Tensor

    """
    image_data = re.sub('^data:image/.+;base64,', '', image)

    image_PIL = Image.open(BytesIO(base64.b64decode(image_data)))

    image_modifications = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Applying transforms on the image
    image_tensor = image_modifications(image_PIL)

    return image_tensor


def predict(image_path, model):
    """
        Parameters:
        image_path : Data URI
        model: Model with loaded parameters and the cnn architecture.

    """
    cell_type = {}
    classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    model.eval()
    image_predict_processed = process_image(image_path)
    processed = image_predict_processed.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(processed)
    prob = torch.exp(output)
    predict_cpu_value = prob
    cell_type['prediction_values'] = [
        prob[0][0].item(), prob[0][1].item(), prob[0][2].item(), prob[0][3].item()]
    predicted_type_index = torch.argmax(predict_cpu_value).item()
    predicted = classes[predicted_type_index]
    if predicted in [classes[1], classes[2]]:
        cell_type['type'] = predicted
        cell_type['cell_category'] = "Mononuclear"
    else:
        cell_type['type'] = predicted
        cell_type['cell_category'] = "Polynuclear"

    return cell_type


# Triggered when we access the home
@app.route("/")
def index():
    return render_template('index.html') # The index.html file should be in the template folder.

# Triggered when we make a POST request. 
@app.route("/predictPM", methods=['POST'])
def predictPM():
    if(request.method == 'POST'):
        img = request.get_json() # Getting the image data object i.e. the data URI.
        a = predict(img['data'], model)
        # print(a)
        return jsonify(a)


if __name__ == '__main__':
    load_model() # Loading the checkpoint only once i.e. when we access the home page.
    app.run(debug=True)
