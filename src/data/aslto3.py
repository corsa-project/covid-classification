import torch
import pandas as pd
import os
import pydicom

from glob import glob
from natsort import natsorted 
from . import dcmtools
from torch.utils.data.dataset import Dataset
from pydicom.dicomdir import DicomDir
from torchvision.datasets.folder import default_loader


class ASLTO3Validation(Dataset):
    def __init__(self, root, transform=None):
        self.loader = default_loader
        self.df = pd.read_csv(os.path.join(root, f"labels.csv"))
        self.df = self.df.rename(columns={'Numero casuale': 'id', 'COVID ': 'covid'})
        self.df = self.df.sort_values(by="id")

        files = glob(os.path.join(root, "png", "*.png"))
        self.files = natsorted(files)
        self.df['image'] = self.files
        self.df.covid = self.df.covid.apply(lambda v: 1 - v)

        self.T = transform
        self.root = root

        print(f"Loaded {len(self.df)} images")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        entry = self.df.iloc[index]
        
        sample = self.loader(self.files[index]).convert("L")
        if self.T is not None:
            sample = self.T(sample)

        # print(index, self.files[index], entry.id)
        return sample, entry.covid


if __name__ == '__main__':
    import argparse
    from torchvision import transforms

    parser = argparse.ArgumentParser(description="Co.R.S.A - covid classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    args = parser.parse_args()

    mean, std = [0.5024], [0.2898]

    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = ASLTO3Validation(args.data_dir, transform=T_test)
    print(test_dataset.df.head())