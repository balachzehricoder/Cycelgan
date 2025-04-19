from PIL import Image
import os
import torch
from torchvision import transforms
from modles.generator import GeneratorResNet

def test(input_dir, output_dir, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # Load Generator
    model = GeneratorResNet(3, 3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"Testing on: {input_dir} ➜ {output_dir}")
    images = os.listdir(input_dir)[:5]

    for img_name in images:
        path = os.path.join(input_dir, img_name)
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image = (output_tensor.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        output_pil = transforms.ToPILImage()(output_image)
        output_pil.save(os.path.join(output_dir, f"fake_{img_name}"))

    print("✅ Done! Output saved in", output_dir)
