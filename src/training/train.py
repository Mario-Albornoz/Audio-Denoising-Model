from datetime import timezone, datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time

from src.evaluation.metrics import  MultiScaleLoss
from src.model.demucs import BaseDemucs
from src.utils.audio_dataset import DenoisingDataSet

def train_model(
        noisy_dir,
        clean_dir,
        epochs=30,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        max_audio_length=80000
):
    dataset = DenoisingDataSet(noisy_dir=noisy_dir, clean_dir=clean_dir, segment_length=max_audio_length)
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
    )

    system = BaseDemucs().to(device)
    optimizer = optim.AdamW(system.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr = 1e-6
    )

    loss_function = MultiScaleLoss()
    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        total_spectral_loss = 0
        total_time_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(loader):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            enhanced = system(noisy)
            loss, time_loss, spectral_loss = loss_function(enhanced, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_spectral_loss += spectral_loss.item()
            total_time_loss += time_loss.item()
            batch_count += 1

            if (batch_idx + 1) % 25 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.6f} - spectral_loss: {spectral_loss.item(): .6f} time_loss: {time_loss.item():.6f}")

            if batch_idx % 2 == 0:
                torch.mps.empty_cache()

        del noisy, clean, enhanced, loss
        torch.mps.empty_cache()
        avg_loss = total_loss / batch_count
        avg_time_loss = total_time_loss / batch_count
        avg_spectral_loss = total_spectral_loss / batch_count
        epoch_time = time.time() - epoch_start

        timestamp = datetime.now(timezone.utc)
        print(
            f"{timestamp.strftime('%H:%M:%S')} | Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f} spectral_loss: {avg_spectral_loss:.6f} time_loss: {avg_time_loss:.6f}|  Time: {epoch_time / 60:.1f}min")

        scheduler.step(avg_loss)
        scheduler.step(avg_spectral_loss)

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
        noisy_dir="../../data/train/noisy",
        clean_dir="../../data/train/clean",
    )
