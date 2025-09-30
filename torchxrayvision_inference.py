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

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import numpy as np

# import sys
# # sys.path.insert(1, '/home/bill/Desktop/projects/concept-explanations/6-cluster')
# sys.path.append('/home/bill/Desktop/projects/concept-explanations/6-cluster')
# from run3 import load_model_c_to_y


def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.apply_sigmoid = False
    model.op_threshs = None
    model.classifier = nn.Linear(model.classifier.in_features, 6)
    model.load_state_dict(torch.load('finetuned_densenet121-res224-all.pth', map_location=torch.device('cpu')))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Model loaded!")
    return model



def prepare_image(img_path: str):
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


def run_model(img_path,model,c_to_y):


    img = prepare_image(img_path)

    with torch.no_grad():
        output = model(img)
        # out = c_to_y(output)
    
    # if isinstance(out, (list, tuple)):
    #     logits = out[0]
    # else:
    #     logits = out
    
    # probs = torch.softmax(logits, dim=1)
    # pred_class = torch.argmax(probs, dim=1).item()


    return output, out, pred_class



# def test_dataset():
#     model = load_model()
#     c_to_y = load_model_c_to_y()
#     # Find the pickle files
#     root = "/home/bill/Desktop/projects/concept-explanations/6-cluster"
#     test_data_path = os.path.join(root, 'test.pkl')

#     with open(test_data_path, "rb") as f:
#         test_data = pickle.load(f)

#         # test_data format: list of dicts or tuples
#         # you need to check what fields exist (commonly ["img_path", "label", "attributes"])
#         print("Example sample:", test_data[0])

#         # ---------------------------
#         # Evaluate
#         # ---------------------------
#         y_true, y_pred = [], []

#         for sample in test_data:
#             # adapt depending on how pickle stores paths + labels
#             if isinstance(sample, dict):
#                 img_path = os.path.join(root, sample["img_path"])
#                 label = sample["class_label"]    # 0/1 cancer label
#             else:
#                 img_path = os.path.join(root, sample[0])  # if tuple
#                 label = sample[1]

#             _, _, pred = run_model(img_path,model,c_to_y)

#             y_true.append(label)
#             y_pred.append(pred)

#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)

#         # ---------------------------
#         # Metrics
#         # ---------------------------
#         acc = accuracy_score(y_true, y_pred)

#         try:
#             auc = roc_auc_score(y_true, y_pred)
#         except ValueError:
#             auc = None  # if only one class present in preds

#         ap = average_precision_score(y_true, y_pred)
#         cm = confusion_matrix(y_true, y_pred)

#         print("Evaluation on test.pkl")
#         print(f"Accuracy: {acc:.4f}")
#         if auc is not None:
#             print(f"ROC-AUC: {auc:.4f}")
#         print(f"PR-AUC: {ap:.4f}")
#         print("Confusion matrix:")
#         print(cm)

if __name__ == '__main__':

    root = "/home/bill/Desktop/projects/concept-explanations/6-cluster/images/"
    img_path = root + os.listdir(root)[0]
    print(img_path)

    output, out, pred_class = run_model(img_path)

    print(output)
    print(type(output))

    print(out)
    print(type(out))

    print(pred_class)

    # test_dataset()