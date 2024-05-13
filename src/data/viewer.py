#  Copyright (c) 2023 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
import pydicom
import argparse
import numpy as np
import matplotlib.pyplot as plt

from . import dcmtools
from pydicom.dicomdir import DicomDir

def show_dataset(ds, indent=""):
    for elem in ds:
        if elem.VR == "SQ":
            indent += 4 * " "
            for item in elem:
                show_dataset(item, indent)
            indent = indent[4:]
        print(indent + str(elem))

def print_metadata(path, ds):
    print()
    print(f"File path...................: {path}")
    print(f"SOP Class...................: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
    print()

    pat_name = ds.PatientName
    print(f"Patient's Name..............: {pat_name.family_name}")
    print(f"Patient ID..................: {ds.PatientID}")
    print(f"Patient Birth...............: {ds.get('PatientBirthDate', '(missing)')}")
    print(f"Modality....................: {ds.Modality}")
    print(f"View Position...............: {ds.ViewPosition}")
    print(f"Study Date..................: {ds.StudyDate}")
    print(f"Image size..................: {ds.get('Rows', '(missing)')} x {ds.get('Columns', '(missing)')}")
    print(f"Pixel Spacing...............: {ds.get('PixelSpacing', '(missing)')}")
    print(f"Slice location..............: {ds.get('SliceLocation', '(missing)')}")
    print(f"Photometric Interpretation..: {ds.get('PhotometricInterpretation', '(missing)')}")
    
    print(f"RescaleSlope................: {ds.get('RescaleSlope', '(missing)')}")
    print(f"RescaleIntercept............: {ds.get('RescaleIntercept', '(missing)')}")
    print(f"WindowCenter................: {ds.get('WindowCenter', '(missing)')}")
    print(f"WindowWidth.................: {ds.get('WindowWidth', '(missing)')}")
    # print("\n\n")
    show_dataset(ds)


def read_dicom(path):
    dcm: DicomDir = pydicom.dcmread(path)
    print_metadata(path, dcm)

    window = dcmtools.get_window(dcm) # dcmtools.get_full_window(dcm)
    lut = dcmtools.make_lut(dcm, *window)
    img = dcmtools.apply_lut(dcm.pixel_array, lut)
    return img
    
"""
Example usage: python3 -m data.viewer --path /path/to/dcm
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Co.R.S.A - dcm viewer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='path of image', required=True)
    opts = parser.parse_args()

    img = read_dicom(opts.path)    

    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()