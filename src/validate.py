import datetime
import os
import torch
import torch.utils.data
import torchvision
import argparse
import time
import wandb
import torch.utils.tensorboard
import data.aslto3 as aslto3
import uuid
import numpy as np

from torchvision import transforms
from util import AverageMeter, BalancedAccuracy, RocAuc, Cutout, ensure_dir, set_seed, arg2bool
from models.resnet import ResNet
from models.hr import HierarchicalResidualFT, HierarchicalResidualFTNorm
from multiprocessing import Process
from fairkl import fairkl



def parse_arguments():
    parser = argparse.ArgumentParser(description="Co.R.S.A - covid classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--weights', type=str, help='pretrained weights', default=None, required=True)

    opts = parser.parse_args()
    return opts

def load_model(opts):
    weights = torch.load(opts.weights, map_location=opts.device)
    model = weights['model']
    return model


@torch.no_grad()
def test(test_loader, model, opts):
    accuracy = BalancedAccuracy()
    auc = RocAuc()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    outputs = []
    labels = []

    model.eval()

    t1 = time.time()
    for idx, (images, covid) in enumerate(test_loader):
        data_time.update(time.time() - t1)
        
        images = images.to(opts.device)
        covid = covid.to(opts.device)
        bsz = images.shape[0]

        logits, feats = model(images)
        
        outputs.append(logits.detach())

        labels.append(covid.detach())

        acc = accuracy(logits, covid)
        auc_score = auc(logits, covid) 

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(test_loader) - idx)

        print(f"Test: [{idx + 1}/{len(test_loader)}]:\t"
                f"BT {batch_time.avg:.3f}\t"
                f"ETA {datetime.timedelta(seconds=eta)}\t"
                f"acc@1 {acc:.3f}\t"
                f"auc {auc_score:.3f}")

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, accuracy.res, auc.res

def main(opts):
    set_seed(0)

    mean, std = [0.5024], [0.2898]

    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = aslto3.ASLTO3Validation(opts.data_dir, transform=T_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, 
                                              num_workers=8, persistent_workers=True)
    
    model = load_model(opts)
    
    run_name = (f"corsa_validation_aslto3_{opts.weights}")
    tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)

    ensure_dir(tb_dir)

    opts.model_class = model.__class__.__name__
    opts.torch_version = torch.__version__
    opts.torchvision_version = torchvision.__version__


    wandb.init(project="corsa", entity="eidos", config=opts, name=run_name, sync_tensorboard=True)
    print('Config:', opts)
    print('Model:', model)
    print('torch version:', opts.torch_version)
    print('torchvision version:', opts.torchvision_version)
    print('Run id:', wandb.run.id)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    
    outputs, accuracy_test, auc_test = test(test_loader, model, opts)
    writer.add_scalar("test/acc@1", accuracy_test, 0)
    writer.add_scalar("test/auc", auc_test, 0)
    print(f"test accuracy {accuracy_test:.2f} test auc {auc_test:.2f}")

    outputs = outputs.detach().cpu()
    outputs = torch.softmax(outputs, dim=1)
    test_dataset.df['prob0'] = outputs[:, 0]
    test_dataset.df['prob1'] = outputs[:, 1]
    print(test_dataset.df)
    test_dataset.df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    opts = parse_arguments()
    main(opts)
    