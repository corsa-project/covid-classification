#  Copyright (c) 2023 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import torch
import random
import numpy as np
import os

from typing import Tuple, Optional
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Accuracy:
    """Evaluates the predictions accuracy given an output `torch.Tensor` and a target `torch.Tensor`.

    Args:
        topk (Tuple[int, ...], optional): top-k accuracy identifiers. E.g. to evaluate both top-1 and top-5 accuracy `topk = (1, 5)`.
    """
    
    def __init__(self, topk: Optional[Tuple[int, ...]] = (1,)) -> None:
        self.topk = topk
        self.outputs = []
        self.targets = []
        self.res = None
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> list:
        """Evaluates the accuracy of the outputs given the targets.

        Args:
            outputs (torch.Tensor): tensor defining a prediction.
            targets (torch.Tensor): tensor defining the targets.

        Returns:
            list: list of top-k accuracy, one for each element of `topk`.

        """
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())

        outputs = torch.cat(self.outputs, dim=0)
        targets = torch.cat(self.targets, dim=0)

        maxk = max(self.topk)
        batch_size = targets.shape[0]
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
    
        self.res = res
        return res
    
class BalancedAccuracy:
    """Evaluates the predictions accuracy given an output `torch.Tensor` and a target `torch.Tensor`.
    """
    def __init__(self) -> None:
        self.outputs = []
        self.targets = []
        self.res = None
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> list:
        """Evaluates the accuracy of the outputs given the targets.

        Args:
            outputs (torch.Tensor): tensor defining a prediction.
            targets (torch.Tensor): tensor defining the targets.

        Returns:
            list: list of top-k accuracy, one for each element of `topk`.

        """
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())

        outputs = torch.cat(self.outputs, dim=0)
        targets = torch.cat(self.targets, dim=0)

      
        _, pred = torch.max(outputs, dim=1)
        res = balanced_accuracy_score(targets.cpu(), pred.cpu())
        self.res = res

        return res
    

class RocAuc:
    """Evaluates the rocauc given an output `torch.Tensor` and a target `torch.Tensor`.
    """
    def __init__(self) -> None:
        self.outputs = []
        self.targets = []
        self.res = None
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> list:
        """Evaluates the accuracy of the outputs given the targets.

        Args:
            outputs (torch.Tensor): tensor defining a prediction (raw logits).
            targets (torch.Tensor): tensor defining the targets.

        Returns:
            list: auc score.

        """
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())

        outputs = torch.cat(self.outputs, dim=0)
        targets = torch.cat(self.targets, dim=0)
      
        outputs = torch.softmax(outputs, dim=1)
        res = roc_auc_score(targets.cpu(), outputs[:, 1].cpu())
        self.res = res

        return res
    
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(0)
        w = img.size(1)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def arg2bool(val):
    if isinstance(val, bool):
        return val
    
    elif isinstance(val, str):
        if val == "true":
            return True
        
        if val == "false":
            return False
    
    val = int(val)
    assert val == 0 or val == 1
    return val == 1