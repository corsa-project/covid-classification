import argparse
import glob
import os
import glob

from util import ensure_dir
from data.preprocess import read_dicom, export_to_png
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Co.R.S.A - data preprocessing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help='path of aslto3 dir', required=True)
    parser.add_argument('--size', type=str, help='image size', default='224x224')
    args = parser.parse_args()
    args.size = [int(s) for s in args.size.split("x")]
    
    ensure_dir(os.path.join(args.input, 'png'))

    for path in tqdm(glob.glob(os.path.join(args.input, "**/*"), recursive=True)):
        if os.path.basename(path) == 'DICOMDIR' or not os.path.isfile(path):
            continue

        if '.TXT' in path or '.XML' in path:
            continue

        if not 'DICOM' in path:
            continue
        
        try:
            dcm = read_dicom(path)
            patient_id = path.split('/')[7]
            print(path, patient_id)
            export_to_png(dcm, args.size, os.path.join(args.input, 'png', patient_id + '_' + os.path.basename(path)))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
