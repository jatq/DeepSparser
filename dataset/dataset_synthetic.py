import numpy as np
import math
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, s, n, config):
        """
        s: clean signal
        n: noise
        config: configuration
        """
        self.config = config
        self.s = self.normalization_1d(self._pad(s))[0] 
        self.n = self.normalization_1d(self._pad(n))[0]

    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    @staticmethod
    def normalization_1d(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        x = (x-mean)/std
        return x, std
    
    def _pad(self, x):
        width, stride, N = self.config.dct_width, self.config.dct_stride, x.shape[-1]
        N_new = math.ceil((N - width) / stride) * stride + width
        if x.ndim == 2:
            return np.pad(x,((0,0),(0, N_new - N)), 'constant')
        return np.pad(x,(0, N_new - N), 'constant')
    
    @staticmethod
    def add_certain_noise(s, n, snr):
        # y = s + k * n, asssume the variance of s and n are both 1, then snr = 10 * lg(1/k**2) = -20 * lg(k), k = 10 ** (-snr/20)
        y = s + n * 10 ** (-snr/20)
        return y
    
    @staticmethod
    def cal_snr(s, n):
        return 10 * np.log10(np.var(s) / np.var(n))



class TrainDataset(MyDataset):
    def __init__(self, s, n, config):
        super(TrainDataset, self).__init__(s, n, config)

    def __getitem__(self, idx):
        s, n = self.s[idx], self.n[np.random.choice(len(self.n))]
        y = self.add_certain_noise(s, n, np.random.choice(np.arange(-10, 10, 0.1)))
        return s, y


class TestDataset(MyDataset):
    def __init__(self, s, n, config):
        super(TestDataset, self).__init__(s, n, config)

    def __getitem__(self, idx):
        s, n = self.s[idx], self.n[idx]
        return s, n
