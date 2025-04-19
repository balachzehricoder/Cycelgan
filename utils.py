from torchvision import transforms
from PIL import Image
import os

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

def load_image(file_path, transform):
    image = Image.open(file_path).convert("RGB")
    return transform(image)
