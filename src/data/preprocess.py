#  Copyright (c) 2023 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import pandas as pd
import os
import pydicom
import argparse
import numpy as np
import datetime
import time

from . import dcmtools
from torch.utils.data.dataset import Dataset
from pydicom.dicomdir import DicomDir
from util import ensure_dir
from tqdm import tqdm
from PIL import Image
from multiprocessing import Manager, Process
from multiprocesspandas import applyparallel

manager = Manager()
processed = manager.Value("processed", 0)

def read_dicom(path):
    dcm: DicomDir = pydicom.dcmread(path)
    window = dcmtools.get_window(dcm) # dcmtools.get_full_window(dcm)
    lut = dcmtools.make_lut(dcm, *window)
    img = dcmtools.apply_lut(dcm.pixel_array, lut)
    return img

def export_to_png(img: np.ndarray, size, path):
    ensure_dir(os.path.dirname(path))

    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize(size)
    pil_img.save(path + ".png")

def process(entry, opts):
    dcm = read_dicom(os.path.join(opts.data_dir, entry.path))
    export_to_png(dcm, opts.size, os.path.join(opts.dest, entry.path))
    processed.value += 1

def clock(df):
    t1 = time.time()
    while processed.value < len(df):
        delta = time.time() - t1
        print("Elapsed time", datetime.timedelta(seconds=delta), f"processed {processed.value}/{len(df)}")
        time.sleep(2)

def main(opts):
    """
        Preprocess dataset and export to png for faster training
    """
    df = pd.read_csv(os.path.join(opts.data_dir, "corda.csv"))
    df = df[df.modality.isin(opts.modalities)]
    
    opts.dest = os.path.join(opts.data_dir, opts.size)
    print("Exporting to", opts.dest)
    print("Total number of images:", len(df))

    p1 = Process(target=clock, args=(df,))
    p1.start()
    
    opts.size = [int(s) for s in opts.size.split("x")]
    df.apply_parallel(process, opts=opts, axis=0)

    p1.join()
    

"""
Example usage: python3 -m data.preprocess --data_dir /home/barbano/data/corda --size 244x244
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Co.R.S.A - data preprocessing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--size', type=str, help='image size', default='224x224')
    parser.add_argument('--modalities', type=str, default="CR,DX")
    opts = parser.parse_args()

    opts.modalities = opts.modalities.split(",")
    main(opts) 