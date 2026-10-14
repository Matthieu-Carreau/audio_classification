from torch.utils.data import Dataset
import librosa
import torchaudio
import os
import numpy as np



class Dataset(Dataset):
    def __init__(self, folder, df_classes):

        self.folder = folder
        list_dir = os.listdir(folder)
        self.files = []
        self.labels = []

        n_classes = len(df_classes)
        label_map = {'NEG': np.zeros(n_classes, dtype=int)}
        for i, label in enumerate(df_classes.Code):
            one_hot_vector = np.zeros(n_classes, dtype=int)
            one_hot_vector[i] = 1
            label_map[label] = one_hot_vector

        for file in list_dir:
            if file[-4:] == ".wav":
                code = file.split("_XC")[0]
                if code in label_map: 
                    # Ignore files with unknown labels
                    self.files.append(file)
                    self.labels.append(label_map[code])

        self.label_map = label_map
        self.labels = np.array(self.labels, dtype=np.float32)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip_path = os.path.join(self.folder, self.files[idx])
        waveform = librosa.load(clip_path, sr=None)[0]
        #waveform = torchaudio.load(clip_path)[0][0]
        label = self.labels[idx]

        return waveform, label