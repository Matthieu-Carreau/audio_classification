# %% [markdown]
# # Détection d’activités d’oiseaux dans des enregistrements audio par réseaux de neurones frugaux
# 
# Pour utiliser un GPU avec google colab : exécution -> modifier le type d'exécution -> GPU

# %% [markdown]
# ## 0 - Si besoin : (re-)télécharger les données audio
# 
# Soit en les téléchargeant et décompressant depuis les liens suivants :
# - examples.zip (4.2 Mo) : https://uncloud.univ-nantes.fr/index.php/s/6XQPXKQ8Syyq3Xb -> pour les deux parties
# - dataset.zip (1.1 Go) : https://uncloud.univ-nantes.fr/index.php/s/CEjynZsiXAX8P8g -> pour la seconde partie
# 
# Soit en lançant la cellule suivante :
# 

# %%
import zipfile, urllib.request

# Tétécharger les fichiers du git (pour les deux parties)
'''
import shutil
from pathlib import Path

# files.zip : https://uncloud.univ-nantes.fr/index.php/s/wyw9sZT3AMD3fCs
url = "https://uncloud.univ-nantes.fr/public.php/dav/files/wyw9sZT3AMD3fCs/?accept=zip"
urllib.request.urlretrieve(url, "files.zip")
with zipfile.ZipFile("files.zip") as z:
    z.extractall("")

for item in Path("files").iterdir():
    shutil.move(str(item), ".")
Path("files").rmdir()
'''

# examples.zip : https://uncloud.univ-nantes.fr/index.php/s/6XQPXKQ8Syyq3Xb -> pour les deux parties
'''
url = "https://uncloud.univ-nantes.fr/public.php/dav/files/6XQPXKQ8Syyq3Xb/?accept=zip"  
urllib.request.urlretrieve(url, "examples.zip")
with zipfile.ZipFile("examples.zip") as z:
    z.extractall("data")
'''

# dataset.zip : https://uncloud.univ-nantes.fr/index.php/s/CEjynZsiXAX8P8g -> pour la partie 2
'''
url = "https://uncloud.univ-nantes.fr/public.php/dav/files/CEjynZsiXAX8P8g/?accept=zip" 
urllib.request.urlretrieve(url, "dataset.zip")
with zipfile.ZipFile("dataset.zip") as z:
    z.extractall("data")
'''


# %% [markdown]
# # Partie 2 - Détection par réseaux de neurones

# %%
import os
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pandas as pd

import librosa
import numpy as np

from src.bcresnet import BCResNets
from src.dataset import Dataset
from src.training import train
from src.utils import plot_mel_spec

# %%
df_classes = pd.read_csv("data/classes.csv", sep=";")
n_classes = len(df_classes)
print(df_classes)

# %% [markdown]
# ## 2.1 - Initialisation d'un BC-ResNet

# %%
bcresnet = BCResNets(base_c=8, 
                     num_classes=n_classes).eval()

# %%
# affichage de la liste des couches
print(bcresnet)

# %% [markdown]
# ### Affichage de certains poids du réseau de neurones

# %%
# Affichage des poids de la première couche de convolution
print(bcresnet.cnn_head[0].weight)

# %%
# Affichage des poids de la dernière couche du réseau de neurones
print(bcresnet.classifier[-1].weight)

# %% [markdown]
# ## 2.3 - Inférence sur un exemple

# %%
# chargement d'un clip audio
base_dir = "data/examples"
f_name = "DENMAJ_D_XC704957_t15.wav"
wav_path = os.path.join(base_dir, f_name)

waveform, rate = librosa.load(wav_path, sr=None)

# %%
# Définition des paramètres du spectrogramme
n_mels = 40

sample_rate = 24000
f_min_mel = 1000
f_max_mel = 12000

hop_ms = 16
hop_length = int(hop_ms * 24)
win_length = 2 * hop_length
n_fft = 1024

eps = 1e-8
log_transform = lambda x: torch.log1p(x / eps)

spectrogram = T.MelSpectrogram(sample_rate=sample_rate, win_length=win_length, n_fft=n_fft, 
                                            hop_length=hop_length, f_min=f_min_mel, f_max=f_max_mel,
                                            n_mels=n_mels, normalized=False, window_fn=torch.hann_window) 


# %%
# Calcul et affichage du spectrogramme
mel_spec = spectrogram(torch.tensor(waveform))
log_mel_spec = log_transform(mel_spec)

plot_mel_spec(log_mel_spec, f_min_mel, f_max_mel, n_mels, 
            hop_length, waveform.shape[-1], sample_rate,
            title=f_name)

log_mel_spec = log_mel_spec.reshape(1, 1, *log_mel_spec.shape)

# %% [markdown]
# ### Logits et scores de confiance
# 
# On obtient la sortie du réseau de neurones avec ```bcresnet(log_mel_spec)```
# Cette sortie est un vecteur de nombres réels quelconques appelés *logits*.
# Ils peuvent être convertis en des scores de confiance compris entre 0 et 1 en appliquant le fonction sigmoïde :
# $\sigma(z) = \frac{1}{1 + e^{-z}}$
# implémentée en pyTorch sous ```torch.sigmoid```

# %%
logits = ...
confidence_scores = ...

print("Logits :")
print(logits.detach().numpy())
print("Scores de confiance :")
print(confidence_scores.detach().numpy())

# %% [markdown]
# ## 2.4 - Utilisation d'un dataset

# %%
train_folder = "data/dataset/train/files"
eval_folder = "data/dataset/eval/files"

train_set = Dataset(train_folder, df_classes)
eval_set = Dataset(eval_folder, df_classes)

# %%
waveform, label = train_set[0]
print(f"Taille de la forme d'onde : {waveform.shape}")
print(f"Label: {label}")

# %%
print("Nombre de clips positifs par classe :")
print(train_set.labels.sum(0))

neg_files = (train_set.labels.max(1) == 0).sum()
print("Nombre de clips négatifs :")
print(neg_files)

# %% [markdown]
# ## 2.5 - Apprentissage du réseau de neurones
# 
# Cette partie est lourde en calcul et peut être passée.
# La partie suivante fournit directement un réseau de neurones entraîné.

# %%
# Paramètres d'apprentissage
batch_size = 32
n_epochs = 2
lr = 1e-3

tau = 2 # Taille du réseau de neurones

# Utilisation de CPU ou GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "mps:0"

seed = 0

torch.manual_seed(seed)
bcresnet = BCResNets(base_c=int(8*tau), 
                     num_classes=n_classes)

hist = train(bcresnet, spectrogram, log_transform, 
          lr, n_epochs, batch_size,
          train_set, eval_set, n_classes, 
          device)

train_loss_hist, train_f1_hist, eval_loss_hist, eval_f1_hist = hist
train_f1_hist = torch.vstack(train_f1_hist).cpu().numpy()
eval_f1_hist = torch.vstack(eval_f1_hist).cpu().numpy()

# %%
plt.plot(train_loss_hist, label="Train")
plt.plot(eval_loss_hist, label="Eval")
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Fonction de coût au cours de l'entraînement")
plt.show()

plt.plot(train_f1_hist.mean(1), label="Train")
plt.plot(eval_f1_hist.mean(1), label="Eval")
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.xlabel("Epochs")
plt.ylabel("F1 score")
plt.title("Moyenne des F1 scores par classe au cours de l'entraînement")
plt.show()


# %%
# Sauvegarde du réseau entraîné
# torch.save(bcresnet, "trained_bcresnet")

# %% [markdown]
# ## 2.6 - Test sur quelques exemples

# %%
# Chargement du réseau de neurones entraîné
loaded_model = torch.load("trained_bcresnet", weights_only=False).to("cpu").eval()
spectrogram.to("cpu")

# %%
threshold = 0.5

examples_folder = "data/examples"
f_names = os.listdir(examples_folder)

for f_name in f_names:
    # Chargement de chaque fichier audio
    wav_path = os.path.join(examples_folder, f_name)
    waveform, rate = librosa.load(wav_path)
    waveform = torch.tensor(waveform)

    # Calcul du spectrogramme
    log_mel_spec = ...

    plt.imshow(log_mel_spec)
    plt.title(f_name)
    plt.show()

    log_mel_spec = log_mel_spec.reshape(1, 1, *log_mel_spec.shape)

    # Compléter pour calculer les sorties du réseau de neurones
    logits = ...
    scores = ...

    # Tri par score décroissant
    scores = scores[0].detach()
    idx = torch.argsort(scores, descending=True).numpy()

    print("Détections:")
    if max(scores) < threshold:
        print("Aucune")

    for i in idx:
        # Compléter avec une condition de détection sur scores[i]
        if ... :
            detected_class = df_classes.loc[i]
            print(f"- {detected_class["Type"]} de {detected_class["Espece"]} ({detected_class["Code"]})", end="\t")
            print(f"Confiance: {float(scores[i])*100:.2f} %")

    print()


