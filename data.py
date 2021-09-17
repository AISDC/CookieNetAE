from torch.utils.data import Dataset
import numpy as np
import h5py, torch, random

class CookieAEDataSet(Dataset):
    def __init__(self, ch=128, ):
        with h5py.File('dataset/Training_set_%dchan.h5' % ch, 'r') as h5fd:
            self.features = h5fd['X_train'][:][:, np.newaxis].astype(np.float32)
            self.targets  = h5fd['Y_train'][:][:, np.newaxis].astype(np.float32)
            self.len = self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return self.len

def get_validation_ds(ch=128, dev='cuda'):
    with h5py.File('dataset/Test_set_%dchan.h5' % ch, 'r') as h5fd:
        features = h5fd['X_test'][:][:, np.newaxis].astype(np.float32)
        targets  = h5fd['Y_test'][:][:, np.newaxis].astype(np.float32)
        return torch.from_numpy(features).to(dev), torch.from_numpy(targets).to(dev)
