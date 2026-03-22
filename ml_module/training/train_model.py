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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("../dataset", transform=transform)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

model = EfficientNet.from_pretrained("efficientnet-b0")

model._fc = nn.Linear(model._fc.in_features, 2)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

all_preds = []
all_labels = []

for epoch in range(5):

    for images, labels in loader:

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().numpy())
        all_labels.extend(labels.detach().numpy())

    print("Epoch", epoch+1, "finished")

accuracy = accuracy_score(all_labels, all_preds)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["fake","real"]))

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

torch.save(model.state_dict(), "../models/deeptrace_model.pth")

print("\nTraining completed")