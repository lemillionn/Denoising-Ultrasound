import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class UltrasoundDataset(Dataset):
    """
    Loads variable-length ultrasound bursts from either .npz archives or single .npy files.
    Returns each example as a tensor of shape [1, T].
    """
    def __init__(self, clean_path, noisy_path, normalize=False):
        # Load clean data (npz archive or single npy array)
        loaded_clean = np.load(clean_path, allow_pickle=True)
        if hasattr(loaded_clean, 'files'):
            self.clean = [loaded_clean[k].astype(np.float32) for k in loaded_clean.files]
        else:
            self.clean = [loaded_clean.astype(np.float32)]

        # Load noisy data
        loaded_noisy = np.load(noisy_path, allow_pickle=True)
        if hasattr(loaded_noisy, 'files'):
            self.noisy = [loaded_noisy[k].astype(np.float32) for k in loaded_noisy.files]
        else:
            self.noisy = [loaded_noisy.astype(np.float32)]

        self.normalize = normalize
        if normalize:
            # Compute global stats once
            all_noisy = np.concatenate(self.noisy)
            mu, sigma = all_noisy.mean(), all_noisy.std()
            self.noisy = [(x - mu) / sigma for x in self.noisy]
            self.clean = [(x - mu) / sigma for x in self.clean]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.noisy[idx])[None]  # shape [1, T]
        y = torch.from_numpy(self.clean[idx])[None]
        return x, y


def collate_fn(batch):
    """
    Pads a batch of variable-length [1, T] tensors to the length of the longest sequence.
    Returns x_pad, y_pad of shape [B, 1, T_max].
    """
    xs, ys = zip(*batch)
    # remove channel dim for padding
    xs_ = [x.squeeze(0) for x in xs]
    ys_ = [y.squeeze(0) for y in ys]
    x_pad = pad_sequence(xs_, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys_, batch_first=True, padding_value=0.0)
    # restore channel dim
    x_pad = x_pad.unsqueeze(1)
    y_pad = y_pad.unsqueeze(1)
    return x_pad, y_pad
