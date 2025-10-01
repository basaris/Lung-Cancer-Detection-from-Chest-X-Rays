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



def load_model():
    """ We load the pre-trained model from https://github.com/mlmed/torchxrayvision"""

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.apply_sigmoid = False   # disable: op_norm
    model.op_threshs = None       #          thresholding
    model.classifier = nn.Linear(model.classifier.in_features, 1)   # change the output of the densenet to 1 (binary classification) instead of 18 
    model = model.to(device)

    return model


def load_dataset(dataset_path):
    """ The current project handles datasets exactly like https://github.com/ml-research/CB2M/tree/main/CUB """

    train_data_path = os.path.join(dataset_path, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    img_root = os.path.join(dataset_path, 'images')

    trainloader = load_data([train_data_path], False, False, 16, image_dir=img_root, n_class_attr=2)
    testloader = load_data([val_data_path], False, False, 16, image_dir=img_root, n_class_attr=2)

    print("Trainloader and testloader loaded correctly!")

    return trainloader, testloader


def run_epochs(model, trainloader, testloader, num_epochs, no_freezing):
    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # Freeze backbone if no_freezing is False.
    # If the dataset is small the best practice is to freeze all the backbone and train only the classifier. After 
    # a number of epochs, when the classifier will be trained, you can unfreeze the backbone to learn more complex patterns.
    # That's what we are doing after the 5th epoch.
    if no_freezing:
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    # Learning rate scheduler to adjust the learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(0, num_epochs):
        if epoch == 5 and not no_freezing:  
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            print("Epoch [5/10], unfreezed the backbone.")

        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")


        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (preds == labels.int()).sum().item()
                val_total   += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {running_loss/len(trainloader):.4f} | "
            f"Val Loss: {val_loss/len(testloader):.4f} | "
            f"Val Acc: {val_correct/val_total:.4f}")

        scheduler.step()

    
    print('Fine-tuning complete!')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'finetuned_densenet121-binary.pth')
    print(f"Model saved on /finetuned_densenet121-binary.pth")


if __name__ == '__main__':

    model = load_model()

    root = "/your/path/to/the/train.pkl/,/val.pkl/and/test.pkl/files/"
    trainloader, testloader = load_dataset(root)

    run_epochs(model=model, trainloader=trainloader, testloader=testloader, num_epochs=10, no_freezing=False)