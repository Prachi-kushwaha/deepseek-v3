import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from mtp import MTP
from model import Deepseekv3_model
from config import ModelConfig , TrainingConfig
from checkpoint import save_checkpoint, load_checkpoint
from data_pipeline import train_loader, val_loader, tokenizer


def compute_mtp_loss(mtp_logits, input_ids, lambda_mtp):
    mtp_losses = []

    for k, logits in enumerate(mtp_logits, start=1):
        length = logits.size(1)
        targets = input_ids[:, k:k + length]

        assert targets.size(1) == length, (
            f"Depth {k}: logits length={length}, "
            f"target length={targets.size(1)}"
        )

        loss_mtp = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        mtp_losses.append(loss_mtp)

    if not mtp_losses:
        return torch.zeros((), device=input_ids.device)

    return lambda_mtp * torch.stack(mtp_losses).mean()



config = ModelConfig()
model = Deepseekv3_model(config)


config = TrainingConfig()

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr_rate,
    betas=config.betas,
    weight_decay=config.weight_decay,
    eps=config.eps,
)


# 2. Generate text using greedy decoding

def generate_text(model,tokenizer, prompt, device, max_new_tokens=30):
    model.eval()

    # Encode using the tokenizers.Tokenizer API
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    context_limit = 128

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            model_input = input_ids[:, -context_limit:]

            main_logits, _, _ = model(model_input)

            next_token_logits = main_logits[:, -1, :]
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            eos_id = tokenizer.token_to_id("</s>")

            if eos_id is not None and next_token.item() == eos_id:
                break

    # Decode using the tokenizers.Tokenizer API
    generated_ids = input_ids[0].tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )



# 3. Training and validation
prompt = "there was a girl called Emily"

def train(model, config):
    num_epochs = config.num_epochs
    lambda_mtp = config.mtp_lambda
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    os.makedirs("checkpoints", exist_ok=True)

    start_epochs = 0
    checkpoint_path = "checkpoints/checkpoint_epoch_2.pt"
    if os.path.exists(checkpoint_path):
        start_epochs = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
    for epoch in range(start_epochs, num_epochs):


       # Training
       model.train()

       train_loss = 0.0
       total_main_loss = 0.0
       total_mtp_loss = 0.0
       total_aux_loss = 0.0

       for x_batch, y_batch in train_loader:

           x_batch = x_batch.to(device)
           y_batch = y_batch.to(device)

           main_logits, mtp_logits, loss_balance = model(x_batch)

           # Main next-token prediction loss.
           main_loss = loss_fn(
               main_logits.reshape(
                   -1, main_logits.size(-1)
               ),
               y_batch.reshape(-1),
           )

           # MTP loss.
           mtp_loss = compute_mtp_loss(
               mtp_logits,
               x_batch,
               lambda_mtp=lambda_mtp,
           )

           # Total objective.
           total_loss = main_loss + mtp_loss + loss_balance

           optimizer.zero_grad(set_to_none=True)
           total_loss.backward()
           optimizer.step()

           train_loss += total_loss.item()
           total_main_loss += main_loss.item()
           total_mtp_loss += mtp_loss.item()
           total_aux_loss += loss_balance.item()

       n_train_batches = len(train_loader)

       avg_train_loss = train_loss / max(n_train_batches, 1)
       avg_main_train = total_main_loss / max(n_train_batches, 1)
       avg_mtp_train = total_mtp_loss / max(n_train_batches, 1)
       avg_aux_train = total_aux_loss / max(n_train_batches, 1)


       # Validation

       model.eval()

       val_loss = 0.0
       total_main_val = 0.0
       total_mtp_val = 0.0
       total_aux_val = 0.0

       with torch.inference_mode():

           for x_batch, y_batch in val_loader:

               x_batch = x_batch.to(device)
               y_batch = y_batch.to(device)

               main_logits, mtp_logits, loss_balance = model(x_batch)

               main_loss = loss_fn(
                   main_logits.reshape(
                       -1, main_logits.size(-1)
                   ),
                   y_batch.reshape(-1),
               )

               mtp_loss = compute_mtp_loss(
                   mtp_logits,
                   x_batch,
                   lambda_mtp=lambda_mtp,
               )

               total_loss = main_loss + mtp_loss + loss_balance

               val_loss += total_loss.item()
               total_main_val += main_loss.item()
               total_mtp_val += mtp_loss.item()
               total_aux_val += loss_balance.item()

       n_val_batches = len(val_loader)

       avg_val_loss = val_loss / max(n_val_batches, 1)
       avg_main_val = total_main_val / max(n_val_batches, 1)
       avg_mtp_val = total_mtp_val / max(n_val_batches, 1)
       avg_aux_val = total_aux_val / max(n_val_batches, 1)

    # log losses

       print(f"\nEpoch {epoch + 1}/{num_epochs} avg train loss = {avg_train_loss} ang val loss = {avg_val_loss}")
       print("-" * 55)

       save_checkpoint(
           model,
           optimizer,
           epoch+1,
           avg_train_loss,
           f"checkpoints/checkpoint_epoch_{epoch + 1}.pt",
       )


        # Save the latest weights separately
       torch.save(
            model.state_dict(),
            "checkpoints/deepseek_v3_weights.pt",
        )



       generated_text = generate_text(
           model=model,
           tokenizer=tokenizer,
           prompt=prompt,
           device=device,
           max_new_tokens=30,
       )

       print("\nGenerated sample:")
       print(generated_text)
       print("=" * 30)



if __name__ == "__main__":
    train(model, config)