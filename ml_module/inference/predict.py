import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained("efficientnet-b0")

model._fc = torch.nn.Linear(model._fc.in_features, 2)

model.load_state_dict(torch.load("../models/deeptrace_model.pth"))

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open("test.jpg")

image = transform(image).unsqueeze(0)

output = model(image)

prediction = torch.argmax(output)

if prediction == 0:
    print("Real Image")
else:
    print("Deepfake Image")