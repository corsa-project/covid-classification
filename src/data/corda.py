import torch
import pandas as pd
import os
import pydicom

from . import dcmtools
from torch.utils.data.dataset import Dataset
from pydicom.dicomdir import DicomDir
from torchvision.datasets.folder import default_loader

institution_id = {
    'molinette': 0,
    'mauriziano': 1,
    'sanluigi': 2,
    'monzino': 3
}

id2institution = {
    0: 'molinette',
    1: 'mauriziano',
    2: 'sanluigi',
    3: 'monzino'
}

class CORDADCM(Dataset):
    def __init__(self, root, transform=None, train=True, institutions=[0,1,2,3], modalities=['CR', 'DX'], split=0):
        if train:
            self.df = pd.read_csv(os.path.join(root, f"fold{split}_train.csv"))
        else:
            self.df = pd.read_csv(os.path.join(root, f"fold{split}_test.csv"))
        
        print("Loading with modalities", modalities, "institutions", institutions, "split", split, f"(train={train})")

        # CR DX CT SR PR
        self.df = self.df[self.df.modality.isin(modalities)]

        # self.df['institution_name'] = self.df.institution
        # self.df.institution = self.df.institution.map(lambda v: institution_id[v])
        self.df = self.df[self.df.institution.isin(institutions)]

        self.df = self.df.fillna(-1)

        self.T = transform
        self.root = root

        print(f"Loaded {len(self.df)} images")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        entry = self.df.iloc[index]
        dcm: DicomDir = pydicom.dcmread(os.path.join(self.root, entry.path))

        window = dcmtools.get_window(dcm) # dcmtools.get_full_window(dcm)
        lut = dcmtools.make_lut(dcm, *window)
        img = dcmtools.apply_lut(dcm.pixel_array, lut)

        if self.T is not None:
            img = self.T(img)

        return img, entry.covid, entry.rx, entry.age, entry.institution
    


class CORDA(CORDADCM):
    def __init__(self, root, transform=None, train=True, institutions=[0,1,2,3], 
                 modalities=['CR', 'DX'], size=(224, 224), loader=default_loader, split=0):
        super().__init__(root, transform=transform, train=train, institutions=institutions, 
                         modalities=modalities, split=split)
        self.root = os.path.join(self.root, 'x'.join(map(str, size)))
        self.loader = default_loader
        print("Using preprocessed images at", self.root)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        entry = self.df.iloc[index]
        
        sample = self.loader(os.path.join(self.root, entry.path) + ".png").convert("L")
        if self.T is not None:
            sample = self.T(sample)

        return sample, entry.covid, entry.rx, entry.age, entry.institution

        