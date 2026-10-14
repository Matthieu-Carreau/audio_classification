import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_f1(tp, fp, fn, eps=1e-8):
    """F1 par classe à partir des comptes cumulés (vecteurs de taille n_classes)."""
    return 2 * tp / (2 * tp + fp + fn + eps)


def run_epoch(model, loader, optimizer, 
              spectrogram, log_transform, 
              n_classes, device, train, threshold=0.5):
    # Sélection du mode (entraînement ou évaluation)
    model.train(train)

    total_loss, n_samples = 0.0, 0
    tp = torch.zeros(n_classes, device=device) # Vrais positifs
    fp = torch.zeros(n_classes, device=device) # Faux positifs
    fn = torch.zeros(n_classes, device=device) # Faux négatifs

    with torch.set_grad_enabled(train):

        # Utilisation du dataloader pour parcourir le dataset
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # Waveform -> spectrogramme de Mel : (B, n_mels, T)
            with torch.no_grad():
                mel_specs = log_transform(spectrogram(waveforms))
            mel_specs = mel_specs.unsqueeze(1)     # (B, 1, n_mels, T) : ajout du canal

            logits = model(mel_specs)          # (B, n_classes)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            if train:
                # Mise à jour des paramètres du réseau de neurones
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Métriques
            total_loss += loss.item() * waveforms.size(0)
            n_samples += waveforms.size(0)

            preds = (torch.sigmoid(logits) > threshold).float()
            tp += (preds * labels).sum(dim=0)
            fp += (preds * (1 - labels)).sum(dim=0)
            fn += ((1 - preds) * labels).sum(dim=0)

    return total_loss / n_samples, compute_f1(tp, fp, fn)


def train(model, spectrogram, log_transform, 
          lr, n_epochs, batch_size,
          train_set, eval_set, n_classes, 
          device):


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    spectrogram = spectrogram.to(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loss_hist, train_f1_hist = [], []
    eval_loss_hist, eval_f1_hist = [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, 
                                         spectrogram, log_transform, 
                                         n_classes, device, train=True)
        eval_loss, eval_f1 = run_epoch(model, eval_loader, optimizer, 
                                       spectrogram, log_transform,
                                       n_classes, device, train=False)

        train_loss_hist.append(train_loss)
        train_f1_hist.append(train_f1)
        eval_loss_hist.append(eval_loss)
        eval_f1_hist.append(eval_f1)


        print(f"Epoch {epoch:02d}/{n_epochs} | "
            f"train loss {train_loss:.4f} | eval loss {eval_loss:.4f} | "
            f"eval F1 moyen {eval_f1.mean():.3f}")
        print("  F1 par classe (eval) :",
            " ".join(f"{v:.2f}" for v in eval_f1.tolist()))

    return train_loss_hist, train_f1_hist, eval_loss_hist, eval_f1_hist