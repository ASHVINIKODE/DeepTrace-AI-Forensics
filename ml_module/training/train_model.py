# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from efficientnet_pytorch import EfficientNet

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])

# dataset = datasets.ImageFolder("../dataset", transform=transform)
# print("Total images:", len(dataset))
# print("Classes:", dataset.classes)

# loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=16,
#     shuffle=True
# )

# model = EfficientNet.from_pretrained("efficientnet-b0")

# model._fc = nn.Linear(model._fc.in_features, 2)

# loss_fn = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for epoch in range(5):

#     for images, labels in loader:

#         outputs = model(images)

#         loss = loss_fn(outputs, labels)

#         optimizer.zero_grad()

#         loss.backward()

#         optimizer.step()

#     print("Epoch finished")

# torch.save(model.state_dict(), "../models/deeptrace_model.pth")

# print("Training completed")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from efficientnet_pytorch import EfficientNet
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from torch.utils.data import random_split, DataLoader

# # ---------------- TRANSFORM ----------------
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])

# # ---------------- DATASET ----------------
# dataset = datasets.ImageFolder("../dataset/image", transform=transform)

# print("Total images:", len(dataset))
# print("Classes:", dataset.classes)

# # ---------------- TRAIN / TEST SPLIT ----------------
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size

# train_data, test_data = random_split(dataset, [train_size, test_size])

# train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# # ---------------- MODEL ----------------
# model = EfficientNet.from_pretrained("efficientnet-b0")
# model._fc = nn.Linear(model._fc.in_features, 2)

# # ---------------- LOSS & OPTIMIZER ----------------
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # ---------------- TRAINING ----------------
# for epoch in range(5):

#     model.train()
#     running_loss = 0

#     for images, labels in train_loader:

#         outputs = model(images)
#         loss = loss_fn(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch+1} finished | Loss: {running_loss:.4f}")

# # ---------------- EVALUATION ----------------
# model.eval()

# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for images, labels in test_loader:

#         outputs = model(images)
#         preds = torch.argmax(outputs, dim=1)

#         all_preds.extend(preds.numpy())
#         all_labels.extend(labels.numpy())

# # ---------------- METRICS ----------------
# accuracy = accuracy_score(all_labels, all_preds)

# print("\n✅ Model Accuracy:", accuracy)

# print("\n📊 Classification Report:\n")
# print(classification_report(all_labels, all_preds, target_names=["fake","real"]))

# cm = confusion_matrix(all_labels, all_preds)

# print("\n📌 Confusion Matrix:")
# print(cm)

# # ---------------- SAVE MODEL ----------------
# torch.save(model.state_dict(), "../models/deeptrace_model.pth")

# print("\n🎉 Training completed & model saved!")

 #just trying 



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import random_split, DataLoader
import os

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- TRANSFORM WITH AUGMENTATION ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
dataset_path = "../dataset/video_frames"  # Use video frames for better generalization
dataset = datasets.ImageFolder(dataset_path, transform=transform)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

# ---------------- TRAIN / TEST SPLIT ----------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# ---------------- MODEL ----------------
model = EfficientNet.from_pretrained("efficientnet-b0")
model._fc = nn.Linear(model._fc.in_features, 2)
model = model.to(device)

# ---------------- LOSS & OPTIMIZER ----------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---------------- TRAINING ----------------
epochs = 5  # reduce if CPU; increase if GPU

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} finished | Loss: {running_loss:.4f}")

# ---------------- EVALUATION ----------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------------- METRICS ----------------
accuracy = accuracy_score(all_labels, all_preds)
print("\n✅ Model Accuracy:", accuracy)

print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["fake","real"]))

cm = confusion_matrix(all_labels, all_preds)
print("\n📌 Confusion Matrix:")
print(cm)

# ---------------- SAVE MODEL ----------------
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/deeptrace_model.pth")
print("\n🎉 Training completed & model saved!")