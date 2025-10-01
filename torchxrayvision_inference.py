import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torchxrayvision as xrv
import skimage
import os
import pickle
from dataset import load_data



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(pth_path):
    """ Pretty straight forward. Load the pretrained model from torchxrayvision, change the output from 18 to 1 just like we did
        when we finetuned the model. After that we load the weights from the .pth file, and ready to go!!!
    """
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.apply_sigmoid = False
    model.op_threshs = None
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(os.path.join(pth_path, 'finetuned_densenet121-binary.pth'), map_location=device))
    model.eval()
    model = model.to(device)

    print("Model loaded!")
    return model



def prepare_image(img_path):
    """Make all the transformations and prepare the image for the model inference"""

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)

    if img.ndim == 3:                 # color image HWC
        img = img.mean(axis=-1)       # or use luminance: img = img[..., :3] @ [0.299,0.587,0.114]
    elif img.ndim == 2:               # already grayscale HW
        pass
    else:
        raise ValueError(f"Unexpected shape {img.shape}")
    
    img = img[None, ...]

    img = transform(img)
    img = torch.from_numpy(img)

    return img[None, ...]


def run_model(model, img_path):
    """ Just run the img through the model and return the output (0 or 1)"""
    img = prepare_image(img_path)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output)   # convert to probability
        pred = (prob > 0.5).int().item() 

    return pred


def pred_func(img_path):
    """ A function to use as a black box. Input a image_url and get a cancer prediction output."""

    pth_path = "/your/path/to/finetuned_densenet121-binary.pth/file"
    model = load_model(pth_path)
    output = run_model(model, img_path)

    return output


if __name__ == '__main__':

    pth_path = "/your/path/to/finetuned_densenet121-binary.pth/file"
    model = load_model(pth_path)

    img_path = "/your/path/to/the/image/ (for example: /home/ubuntu/my_img.jpg)"
    output = run_model(model, img_path)

    if output:
        print("The patient has cancer")
    else:
        print("The patient is healthy!")
