import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataload.collate import collate_fn
from dataload.dataset import SpeechDataset
from models.wav2vec_vc import Wav2vecVC


def train(
        feature_dir,
        features,
        layer_weight_path,
        batch_size,
        n_workers,
        learning_rate,
        lr_sched_steps,
        lr_sched_gamma,
        progress_batches,
        total_steps,
        ckpt_save_steps,
        ckpt_save_dir
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeechDataset(
        feature_dir=feature_dir,
        features_to_use=features
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = Wav2vecVC(layer_weight_path).to(device)

    criterion = nn.L1Loss()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=lr_sched_steps, gamma=lr_sched_gamma)

    progress = tqdm(total=progress_batches * train_loader.batch_size, ncols=0, desc="Train")

    losses = []

    for step in range(total_steps):
        try:
            data = next(train_iterator)
        except:
            train_iterator = iter(train_loader)
            data = next(train_iterator)

        mel = data['mel'].float().to(device)
        w2v = [layer_hs.float().to(device) for layer_hs in data['w2v']]
        padding_mask = data['padding_mask'].to(device)

        dec = model(w2v, padding_mask)

        mel = ~padding_mask.unsqueeze(2) * mel
        dec = ~padding_mask.unsqueeze(2) * dec

        loss_rec = criterion(mel, dec)
        losses.append(loss_rec.item())

        loss = loss_rec

        progress.set_postfix(step=step + 1, loss_rec=loss_rec.item(), train_loss=sum(losses) / len(losses))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=5)
        optimizer.step()
        scheduler.step()

        progress.update(train_loader.batch_size)

        if (step + 1) % progress_batches == 0:
            losses = []
            progress.close()
            progress = tqdm(total=progress_batches * train_loader.batch_size, ncols=0, desc="Train")

        if (step + 1) % ckpt_save_steps == 0:
            os.makedirs(ckpt_save_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_save_dir, f"wav2vec-vc-{step+1}.pt")

            torch.save(model.cpu().state_dict(), ckpt_path)

            model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('-f', '--feature_path', type=str)
    parser.add_argument('-o', '--ckpt_save_path', type=str)
    parser.add_argument('-s', '--speaker_layer_weight_path', type=str)
    parser.add_argument('-c', '--content_layer_weight_path', type=str)

    args = parser.parse_args()

    feature_path = args.feature_path
    ckpt_save_path = args.ckpt_save_path
    layer_weight_path = {
        'speaker': args.speaker_layer_weight_path,
        'content': args.content_layer_weight_path
    }

    train(feature_dir=feature_path,
          features=['mel', 'w2v'],
          layer_weight_path=layer_weight_path,
          batch_size=32,
          n_workers=16,
          learning_rate=0.0005,
          lr_sched_steps=40000,
          lr_sched_gamma=0.5,
          progress_batches=1000,
          total_steps=200000,
          ckpt_save_steps=10000,
          ckpt_save_dir=ckpt_save_path
          )

