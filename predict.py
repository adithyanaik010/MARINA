import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# MODEL CLASS
# -----------------------------
class WeldResNet(nn.Module):

    def __init__(self):
        super(WeldResNet, self).__init__()

        self.model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # 4 classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            4
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    "crack",
    "lack_of_penetration",
    "no_defect",
    "porosity"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
model = WeldResNet().to(device)

model.load_state_dict(
    torch.load("weld_model.pth")
)

model.eval()

print("Model loaded successfully!")

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# IMAGE PATH
# -----------------------------
image_path = r"C:/Users/P ADITHYA M NAIK/Desktop/WeldDefectAI/dataset/train/porosity/bam5_Img2_A80_S3_[1][12].png"
# Open image
image = Image.open(image_path).convert("RGB")

# Keep original for display
display_image = image

# Transform image
image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

# Move to device
image_tensor = image_tensor.to(device)

# -----------------------------
# PREDICTION
# -----------------------------
with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

predicted_class = class_names[
    predicted.item()
]

confidence_score = confidence.item() * 100

# -----------------------------
# RESULT
# -----------------------------
print("\nPrediction Result")
print("------------------")
print(f"Defect Type: {predicted_class}")
print(f"Confidence: {confidence_score:.2f}%")

# -----------------------------
# DISPLAY IMAGE
# -----------------------------
plt.imshow(display_image)
plt.title(
    f"{predicted_class} ({confidence_score:.2f}%)"
)
plt.axis("off")
plt.show()
