# ===============================
# MNIST & Fashion-MNIST
# VAE vs GAN - TEK DOSYA
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# -------------------------------
# DATASET
# -------------------------------
transform = transforms.ToTensor()

def get_dataloader(dataset_name, batch_size=128):
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    else:
        dataset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===============================
# VAE
# ===============================
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.fc3(torch.relu(self.fc2(z))))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def train_vae(dataloader, epochs=10):
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"VAE Epoch {epoch+1}: Loss={avg_loss:.2f}")

    return model, losses

# ===============================
# GAN
# ===============================
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_gan(dataloader, epochs=10, z_dim=100):
    G = Generator(z_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=2e-4)
    opt_D = optim.Adam(D.parameters(), lr=2e-4)
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        for x, _ in dataloader:
            x = x.to(device)
            bs = x.size(0)

            real = torch.ones(bs, 1).to(device)
            fake = torch.zeros(bs, 1).to(device)

            # Discriminator
            z = torch.randn(bs, z_dim).to(device)
            fake_img = G(z)
            d_loss = criterion(D(x), real) + criterion(D(fake_img.detach()), fake)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Generator
            g_loss = criterion(D(fake_img), real)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        print(f"GAN Epoch {epoch+1}: D={d_loss.item():.3f} G={g_loss.item():.3f}")

    return G, g_losses, d_losses

# ===============================
# PLOTS & IMAGES
# ===============================
def plot_loss(losses, title, ylabel):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_gan_losses(g_losses, d_losses, title):
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

def show_generated(generator):
    z = torch.randn(16, 100).to(device)
    imgs = generator(z).detach().cpu()

    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i][0], cmap="gray")
        ax.axis("off")
    plt.show()

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    for dataset_name in ["MNIST", "FashionMNIST"]:
        print("\n==============================")
        print("Dataset:", dataset_name)
        print("==============================")

        loader = get_dataloader(dataset_name)

        # VAE
        vae_model, vae_losses = train_vae(loader, epochs=10)
        plot_loss(
            vae_losses,
            f"VAE Loss ({dataset_name})",
            "Loss"
        )

        # GAN
        gan_G, g_losses, d_losses = train_gan(loader, epochs=10)
        plot_gan_losses(
            g_losses,
            d_losses,
            f"GAN Loss ({dataset_name})"
        )

        print(dataset_name, "GAN örnekleri:")
        show_generated(gan_G)
