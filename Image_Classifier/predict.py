import argparse
import numpy as np
import pandas as pd
import torch
import json
import torchvision
from torch import nn
from collections import OrderedDict
from PIL import Image

def main():
    input = get_arguments()
    
    path_to_image = input.image_path
    gpu = input.gpu
    device = get_device(gpu)
    
    with open(input.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_and_build_checkpoint(input.checkpoint)
    model.to(device)
    predict(path_to_image, model, cat_to_name, device, input.top_k)

def load_and_build_checkpoint(path):
    checkpoint = torch.load(path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_units'], bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=checkpoint['drop'])),
        ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'], bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    return model
    
def get_device(gpu):
    device_type = "cuda:0" if torch.cuda.is_available() and gpu else "cpu"
    print("Device Type: {}".format(device_type))
    return torch.device(device_type)

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, default="flowers/test/10/image_07090.jpg")
    parser.add_argument("checkpoint", type=str, default="cmd_line_checkpoint.pth")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category_names", type=str, default="cat_to_name.json")
    parser.add_argument("--gpu", action='store_true')
    
    return parser.parse_args()

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint['arch'])
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_units'], bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=checkpoint['drop'])),
        ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'], bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.classifier.load_state_dict(checkpoint['state_dict'])
    # model.classifier.optimizer = checkpoint['optimizer']
    # model.classifier.epochs = checkpoint['epochs']
    # model.classifier.learning_rate = checkpoint['learning_rate']

    return model


def process_image(image):
    image = Image.open(image)
    w, h = image.size
    
    if w < h: 
        image.thumbnail((256, 99999999999999))
    else: 
        image.thumbnail((99999999999999, 256))
                        
    left = (image.width-224)/2
    bottom = (image.height-224)/2
    right = left + 224
    top = bottom + 224
    
    image = image.crop((
        left, 
        bottom, 
        right,    
        top
    ))
    
    image = np.array(image)/255
    
    image = (image-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    
    image = image.transpose(2, 0, 1)
    
    return image

def predict(image_path, model, cat_to_name, device, topk=5):
    model.to(device)
    image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).to(device)
    input_tensor = image.unsqueeze(0)
    
    model.eval();
    probs = torch.exp(model.forward(input_tensor))
    
    probabilities = probs.topk(topk)
    
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = probabilities[0].detach().cpu().numpy().tolist()[0] 


    i=0
    while i < topk:
        print("{}: {} Has a probability of {}".format(i+1, labels[i], probability[i]))
        i += 1

if __name__ == "__main__":
    main()