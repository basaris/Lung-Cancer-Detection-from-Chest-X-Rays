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



model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.apply_sigmoid = False   # 🔥 disable op_norm / thresholding
model.op_threshs = None 
model.classifier = nn.Linear(model.classifier.in_features, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Find the pickle files
root = "/home/bill/Desktop/projects/concept-explanations/6-cluster"
train_data_path = os.path.join(root, 'train.pkl')
val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

# # Prepare the image:
img_root = "/home/bill/Desktop/projects/concept-explanations/6-cluster"


trainloader = load_data([train_data_path], False, False, 16, image_dir=img_root, n_class_attr=2)
testloader = load_data([val_data_path], False, False, 16, image_dir=img_root, n_class_attr=2)

print(type(trainloader))
print(type(testloader))

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Only optimize classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# Learning rate scheduler to adjust the learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



# Fine-tune the model
num_epochs = 10
#set num_epochs to a smaller number like 1 and use T4 GPU, wait for 4 minute may be, if training is taking too long in colab. 
for epoch in range(0, num_epochs):
    if epoch == 5:  
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        # attributes = [i.long() for i in attributes]
        # attributes = torch.stack(attributes).t()#.float() #N x 312
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
            # if isinstance(attributes, list):
            #     attributes = [i.long() for i in attributes]
            #     attributes = torch.stack(attributes).t()#.float() #N x 312
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

    # model.eval()
    # val_correct, val_total = 0, 0
    # with torch.no_grad():
    #     for images, labels, attributes in testloader:
    #         attributes = [i.long() for i in attributes]
    #         attributes = torch.stack(attributes).t()#.float() #N x 312
    #         images, attributes = images.to(device), attributes.to(device).float()
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, 1)
    #         val_correct += (predicted == attributes).sum().item()
    #         val_total += attributes.size(0)
    # val_acc = val_correct / val_total

    # print(f"Epoch {epoch+1} | Val Accuracy: {val_acc:.4f}")

    # Step the scheduler after each epoch
    scheduler.step()

    

print('Fine-tuning complete!')

# Save the fine-tuned model
torch.save(model.state_dict(), 'finetuned_densenet121-binary.pth')
print('Model saved!')

# # Set the model to evaluation mode
# model.eval()

# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the fine-tuned model on the test images: {100 * correct / total:.2f}%')