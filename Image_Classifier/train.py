import numpy as np
import torch
import PIL
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import argparse
from PIL import Image


DROPOUT = 0.2
MODELS = {
    "vgg16": {
        "model": models.vgg16(pretrained=True),
        "size": 25088 
    },
    "resnet18": {
        "model": models.resnet18(pretrained=True),
        "size": 512
    },
    "alexnet": {
        "model": models.alexnet(pretrained=True),
        "size": 9216
    }
}

def main():
    input = get_arguments()
    
    data_dir = input.data_directory
    save_to = input.save_dir
    pretrained_model = input.arch
    learning_rate = input.learning_rate
    ep = input.epochs
    hidden_units = input.hidden_units
    output_size = input.output
    
    gpu = input.gpu
    device = get_device(gpu)
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    trainloader, validloader, testloader = get_loaders(train_dir, valid_dir, test_dir)
    
    model = MODELS[pretrained_model]["model"]
    input_size = MODELS[pretrained_model]["size"]
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=DROPOUT)),
        ('fc2', nn.Linear(hidden_units, output_size, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device);

    train(model, classifier, trainloader, testloader, validloader, criterion, optimizer, ep, device)

    validation(testloader, device, model)
    predict("flowers/test/10/image_07090.jpg", model)
    checkpoint = {'input_size': input_size,
        'output_size': output_size,
        'hidden_units': hidden_units,
        'drop': DROPOUT,
        'epochs': ep,
        'learning_rate': learning_rate,
        'arch': pretrained_model,
        'optimizer': optimizer,
        'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, save_to)

def predict(image_path, model, topk=5):
    model.to("cpu")
    image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).to("cpu")
    input_tensor = image.unsqueeze(0)
    
    model.eval();
    probs = torch.exp(model.forward(input_tensor))
    
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    print(top_labs)
    # Convert indices to classes
    idx_to_class = {val: key for key, val in   #                                       model.class_to_idx.items(

    top_labels = [idx_to_class[lab] for lab in top_bs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in topabs]
    print(top_labels, top_flowers)
#     return top_probs, top_labels, top_flowers

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


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, default="vgg16",
            help="pre-trained model options: vgg16, resnet18, alexnet")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden_units", type=int, default=4096)

    parser.add_argument("--gpu", type=bool, action='store_true')
    parser.add_argument("--output", type=int, default=102)

    parser.add_argument("data_directory", type=str, help="directory containing training and testing data")
    parser.add_argument("--save_dir", type=str, default="cmd_line_checkpoint.pth")

    return parser.parse_args()

def get_device(gpu):
    device_type = "cuda:0" if torch.cuda.is_available() and gpu else "cpu"
    print("Device Type: {}".format(device_type))
    return torch.device(device_type)

def get_loaders(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)

    return trainloader, validloader, testloader


def loss_accuracy(model, testloader, criterion, device="cpu"):
    test_loss = 0
    accuracy = 0
    
    for idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        ps = torch.exp(output)
        test_loss += criterion(output, labels).item()
        
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    return test_loss, accuracy

def validation(testloader, device, model):
    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy: {}%'.format(100 * correct / total))

def train(model, classifier, trainloader, testloader, validloader, criterion, optimizer, epochs, device):
    steps = 0
    print_every = 30
    model.classifier = classifier
    model.to(device);
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    for e in range(epochs):
        running_loss = 0
    
        for idx, (inputs, labels) in enumerate(trainloader):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = loss_accuracy(model, validloader, criterion, device)
            
                print("Epoch: {}".format(e))
                print("Training Loss: {}".format(running_loss/print_every))
                print("Validation Loss: {}".format(valid_loss/len(testloader)))
                print("Validation Accuracy: {}".format(accuracy/len(testloader)))    
        
                running_loss = 0
                model.train()

if __name__ == "__main__":
    main()