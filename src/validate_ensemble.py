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
import copy

from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from util import AverageMeter, BalancedAccuracy, RocAuc, Cutout, ensure_dir, set_seed, arg2bool
from models.resnet import ResNet
from models.hr import HierarchicalResidualFT, HierarchicalResidualFTNorm
from multiprocessing import Process
from fairkl import fairkl
from glob import glob
from natsort import natsorted


def parse_arguments():
    parser = argparse.ArgumentParser(description="Co.R.S.A - covid classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--weights', type=str, help='pretrained weights (split=*)', default=None, required=True)

    opts = parser.parse_args()
    return opts

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

    return outputs, labels, accuracy.res, auc.res

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
    
    models = []
    for weights in natsorted(glob(opts.weights)):
        print("=> Loading", weights)
        weights = torch.load(weights, map_location=opts.device)
        models.append(weights['model'])

    # https://arxiv.org/pdf/2203.05482.pdf
    model_soup = copy.deepcopy(models[0]) 
    sd_soup = model_soup.state_dict()
    sd = [m.state_dict() for m in models]
    for key in sd_soup:
        sd_soup[key] = sum([sd[i][key] for i in range(len(sd))]) / len(sd)
    model_soup.load_state_dict(sd_soup)
    
    run_name = (f"corsa_validation_aslto3_{opts.weights}")
    tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)

    ensure_dir(tb_dir)

    opts.model_class = models[0].__class__.__name__
    opts.torch_version = torch.__version__
    opts.torchvision_version = torchvision.__version__


    wandb.init(project="corsa", entity="eidos", config=opts, name=run_name, sync_tensorboard=True)
    print('Config:', opts)
    print('Model:', models[0])
    print('torch version:', opts.torch_version)
    print('torchvision version:', opts.torchvision_version)
    print('Run id:', wandb.run.id)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    
    all_outputs = []
    labels = None

    for split, model in enumerate(models):
        outputs, labels, accuracy_test, auc_test = test(test_loader, model, opts)
        print(f"Split {split} - test accuracy {accuracy_test:.4f} test auc {auc_test:.4f}")
        all_outputs.append(outputs.detach())
    
    all_outputs = torch.stack(all_outputs, dim=0).mean(dim=0)
    
    _, pred = torch.max(all_outputs, dim=1)
    bacc = balanced_accuracy_score(labels.cpu(), pred.cpu())

    outputs = torch.softmax(all_outputs, dim=1)
    rocauc = roc_auc_score(labels.cpu(), all_outputs[:, 1].cpu())
    
    writer.add_scalar("test/acc@1", bacc, 0)
    writer.add_scalar("test/auc", rocauc, 0)
    print("=> Testing ensemble")
    print(f"Ensemble (avg. pred) - test accuracy {bacc:.4f} test auc {rocauc:.4f}")

    outputs = outputs.detach().cpu()
    outputs = torch.softmax(outputs, dim=1)
    test_dataset.df['prob0'] = outputs[:, 0]
    test_dataset.df['prob1'] = outputs[:, 1]
    # print(test_dataset.df)
    test_dataset.df.to_csv('results.csv', index=False)


    print("=> Testing soup") 
    outputs, labels, accuracy_test, auc_test = test(test_loader, model_soup, opts)
    print(f"Soup - test accuracy {accuracy_test:.4f} test auc {auc_test:.4f}")
    writer.add_scalar("test/acc@1", accuracy_test, 0)
    writer.add_scalar("test/auc", auc_test, 0)


if __name__ == '__main__':
    opts = parse_arguments()
    main(opts)
    