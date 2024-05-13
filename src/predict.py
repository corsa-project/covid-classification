import datetime
import os
import torch
import torch.utils.data
import torchvision
import argparse
import time
import wandb
import torch.utils.tensorboard
import pydicom
import json

from torchvision import transforms
from util import AverageMeter, BalancedAccuracy, RocAuc, ensure_dir, set_seed
from models.resnet import ResNet
from models.hr import HierarchicalResidual, HierarchicalResidualFT, HierarchicalResidualFTNorm, chexpert_classes
from pydicom.dicomdir import DicomDir
from data import dcmtools
from torchvision.datasets.folder import default_loader
from pprint import pprint


def parse_arguments():
    parser = argparse.ArgumentParser(description="Co.R.S.A - covid classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--pre_weights', type=str, help='chexpert pretrained weights', default=None, required=True)
    parser.add_argument('--clf_weights', type=str, help='classifier weights', default=None, required=True)
    
    parser.add_argument('--input', type=str, help='input image path (DICOM or png)', required=True)
    parser.add_argument('--output', type=str, help='output prediction file path', default="result.json")

    opts = parser.parse_args()
    
    if not torch.cuda.is_available():
        opts.device = "cpu"
    
    return opts

def load_models(opts):
    pre_weights = torch.load(opts.pre_weights, map_location=opts.device)
    pre_model = HierarchicalResidual(encoder='densenet121', pretrained=False)
    pre_model.load_state_dict(pre_weights['model'])
    pre_model = pre_model.to(opts.device)

    clf_weights = torch.load(opts.clf_weights, map_location=opts.device)
    clf_model = clf_weights['model'].to(opts.device)
    return pre_model, clf_model


@torch.no_grad()
def run_inference(pre_model, clf_model, opts):
    t0 = time.time()

    print("=> Loading image")
    if "png" in opts.input:
        input = default_loader(opts.input).convert("L")
    else:
        dcm: DicomDir = pydicom.dcmread(opts.input)
    t_load = time.time() - t0

    t1 = time.time()

    if "png" in opts.input:
        pass
    else:
        window = dcmtools.get_window(dcm)
        lut = dcmtools.make_lut(dcm, *window)
        input = dcmtools.apply_lut(dcm.pixel_array, lut)
    
    # Transform input
    mean, std = [0.5024], [0.2898]
    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transformed = T_test(input)
    transformed = transformed.unsqueeze(0) # Add batch dimension
    transformed = transformed.to(opts.device)
    t_preproc = time.time() - t1
    
    t1 = time.time()
    pre_model.eval()
    clf_model.eval()
    pre_output = torch.sigmoid(pre_model(transformed))
    clf_output = torch.softmax(clf_model(transformed)[0], dim=1)
    t_inference = time.time() - t1

    radiology_report = {}
    for class_name, output in zip(chexpert_classes, pre_output[0]):
        radiology_report[class_name] = output.item()
    
    result = {
        "input": opts.input,
        "pre_weights": opts.pre_weights,
        "clf_weights": opts.clf_weights,
        "radiology_report": radiology_report,
        "covid_prob": clf_output[0][1].item(),
        "perf": {
            "tot_time": str(datetime.timedelta(seconds=time.time() - t0)),
            "load_time": str(datetime.timedelta(seconds=t_load)),
            "preproc_time": str(datetime.timedelta(seconds=t_preproc)),
            "inference_time": str(datetime.timedelta(seconds=t_inference))
        }
    }

    return result

def main(opts):
    set_seed(0)
    
    print("=> Loading pretrained models")
    pre_model, clf_model = load_models(opts)
    ensure_dir(os.path.dirname(opts.output))    

    print("=> Running inference")
    output = run_inference(pre_model, clf_model, opts)

    print("=> Results:")
    pprint(output)

    print("=> Saving output to", opts.output)
    with open(opts.output, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    opts = parse_arguments()
    main(opts)
    