import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)



def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    previous_loss = checkpoint["loss"]

    print(f"Resuming after epoch {start_epoch}")
    print(f"Previous loss: {previous_loss:.4f}")

    return start_epoch

