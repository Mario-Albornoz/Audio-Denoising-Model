import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.model.conv_tasnet import ConvTasNet
from src.model.diffusion_prior import DiffusionPrior
from src.utils.audio_dataset import DenoisingDataSet


class DenoiseSystem(nn.Module):
    def __init__(self, use_diffusion: bool = True):
        super().__init__()
        self.model = ConvTasNet()
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.diffusion = DiffusionPrior()
        else:
            self.diffusion = None


    def forward(self, noisy:str):
        enhanced = self.model(noisy)

        if not self.use_diffusion:
            return enhanced

        noise_est = self.diffusion(enhanced, t=0)
        return  enhanced - 1 * noise_est

class ImprovedDenoiseSystem(nn.Module):
   def __init__(self):
       super().__init__()
       self.model = ConvTasNet(N=512, L=20, B=512, P=3, X=8, R=4)

   def forward(self, noisy):
       return self.model(noisy)

def train_model(
        noisy_dir,
        clean_dir,
        epochs=20,
        batch_size=5,
        lr=1e-4,
        device=("mps" if torch.backends.mps.is_available() else "cpu")
):
    dataset = DenoisingDataSet(noisy_dir, clean_dir)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)



    #system = ImprovedDenoiseSystem().to(device)
    system = DenoiseSystem(use_diffusion=False).to(device)

    optimizer = optim.AdamW(system.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    loss_function = nn.L1Loss()

    best_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            enhanced = system(noisy)
            loss = loss_function(enhanced, clean)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(system.state_dict(), "best_model.pt")
            print(f"  ✓ New best model saved (loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training complete! Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_model(
        noisy_dir="../../data/processed",
        clean_dir="../../data/raw",
        epochs=10,
        batch_size=4,
        lr=1e-4
    )
