def train():
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from modles.desclamintator import Discriminator
    from modles.generator import GeneratorResNet
    from utils import get_transform
    from tqdm import tqdm
    import itertools
    from PIL import Image
    import os

    class CustomDataset(Dataset):
        def __init__(self, folder_path, transform=None):
            self.folder_path = folder_path
            self.transform = transform
            self.image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img

    batch_size = 1
    input_nc = 3
    output_nc = 3
    n_residual_blocks = 9
    lr = 0.0002
    beta1 = 0.5
    epochs = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G_AB = GeneratorResNet(input_nc, output_nc, n_residual_blocks).to(device)
    G_BA = GeneratorResNet(input_nc, output_nc, n_residual_blocks).to(device)
    D_A = Discriminator(input_nc).to(device)
    D_B = Discriminator(input_nc).to(device)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()

    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    transform = get_transform()

    dataset_A = CustomDataset('datasets/trainA', transform=transform)
    dataset_B = CustomDataset('datasets/trainB', transform=transform)

    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        pbar = tqdm(zip(loader_A, loader_B), desc=f'Epoch {epoch+1}/{epochs}', total=min(len(loader_A), len(loader_B)))

        for i, (real_A, real_B) in enumerate(pbar):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            optimizer_D_A.zero_grad()
            real_validity_A = D_A(real_A)
            fake_A = G_BA(real_B).detach()
            fake_validity_A = D_A(fake_A)
            valid = torch.ones_like(real_validity_A).to(device)
            fake = torch.zeros_like(fake_validity_A).to(device)
            loss_D_A = (criterion_GAN(real_validity_A, valid) + criterion_GAN(fake_validity_A, fake)) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            real_validity_B = D_B(real_B)
            fake_B = G_AB(real_A).detach()
            fake_validity_B = D_B(fake_B)
            loss_D_B = (criterion_GAN(real_validity_B, valid) + criterion_GAN(fake_validity_B, fake)) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            optimizer_G.zero_grad()
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, valid)

            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, valid)

            rec_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A)

            rec_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)

            loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * (loss_cycle_A + loss_cycle_B)
            loss_G.backward()
            optimizer_G.step()

            pbar.set_postfix({
                'D_A': loss_D_A.item(),
                'D_B': loss_D_B.item(),
                'G': loss_G.item()
            })

        torch.save(G_AB.state_dict(), f'G_AB_epoch_{epoch+1}.pth')
        torch.save(G_BA.state_dict(), f'G_BA_epoch_{epoch+1}.pth')
        torch.save(D_A.state_dict(), f'D_A_epoch_{epoch+1}.pth')
        torch.save(D_B.state_dict(), f'D_B_epoch_{epoch+1}.pth')

    print("✅ Training Finished!")
