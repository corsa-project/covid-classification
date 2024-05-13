import datetime
import os
import torch
import torch.utils.data
import torchvision
import argparse
import time
import wandb
import torch.utils.tensorboard
import data.corda as corda
import uuid
import numpy as np
import copy

from torchvision import transforms
from util import AverageMeter, BalancedAccuracy, RocAuc, Cutout, ensure_dir, set_seed, arg2bool
from models.resnet import ResNet
from models.hr import HierarchicalResidualFT, HierarchicalResidualFTNorm, HierarchicalResidualFTNorm2
from multiprocessing import Process
from fairkl import fairkl



def parse_arguments():
    parser = argparse.ArgumentParser(description="Co.R.S.A - covid classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--split', type=int, help='data split', choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--train_sources', type=str, help='list of train institutions (comma separated)', default="0,1,2,3")
    parser.add_argument('--test_sources', type=str, help='list of test insitutions (comma separated)', default="0,1,2,3")
    parser.add_argument('--modalities', type=str, help='imaging modalities (comma separated)', default='DX,CR')
    parser.add_argument('--size', type=str, help='image size', default='244x244')

    parser.add_argument('--model', type=str, help='model architecture', default="resnet18")
    parser.add_argument('--load_weights', type=str, help='load pretrained weights', default=None)
    parser.add_argument('--freeze_encoder', action='store_true', help='freeze encoder')
    parser.add_argument('--restore', type=str, help='restore run', default=None)

    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step', 'none'], default='step')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="33,66")
    parser.add_argument('--optimizer', type=str, help="optimizer (adam, sgd)", choices=["adam", "sgd"], default="sgd")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--test_freq', type=int, help='test frequency', default=1)

    # choices 'none', 'crop', 'cutout', 'rotate'
    parser.add_argument('--aug', type=str, help='data augmentations (comma separated)', default='crop')
    parser.add_argument('--cv', action='store_true', help='do cross validation (4 folds)')
    parser.add_argument('--amp', action='store_true', help='use amp')

    # loss options
    parser.add_argument('--alpha', type=float, help='cross entropy weight', default=1.0)
    parser.add_argument('--lambd', type=float, help='fairkl weight', default=0.)

    parser.add_argument('--same_init', action="store_true", help='use same weights init')
    parser.add_argument('--finetune_conv', action="store_true", help='finetune last conv block')
    parser.add_argument('--fc_reg', action='store_true', help='put fairkl between FC layers')

    parser.add_argument('--prefix', type=str, help='run prefix', default='')

    opts = parser.parse_args()
    if opts.load_weights is not None and opts.restore is not None:
        print("Ony one between --load_weights and --restore must be not None")
        exit(1)
    
    opts.size = [int(s) for s in opts.size.split("x")]
    opts.train_sources = [int(s) for s in opts.train_sources.split(",")]
    opts.test_sources = [int(s) for s in opts.test_sources.split(",")]
    opts.modalities = opts.modalities.split(",")
    opts.aug = opts.aug.split(",")
    return opts

def load_data(opts):
    mean, std = [0.5024], [0.2898]

    augmentations = []
    for aug in opts.aug:
        if aug == "none":
            continue

        if aug == "crop":
            augmentations.append(transforms.RandomResizedCrop(size=opts.size, scale=(0.95, 1.0), antialias=True))
        elif aug == "cutout":
            augmentations.append(Cutout(n_holes=1, length=0.1*opts.size[0]))
            # augmentations.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 0.33)))
        elif aug == "rotate":
            augmentations.append(transforms.RandomRotation(degrees=20))

    T_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Compose(augmentations),
        transforms.Normalize(mean, std)
    ])

    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = corda.CORDA(opts.data_dir, transform=T_train, train=True,
                                institutions=opts.train_sources, 
                                modalities=opts.modalities, size=opts.size, 
                                split=opts.split)
    
    test_dataset = corda.CORDA(opts.data_dir, transform=T_test, train=False,
                                institutions=opts.test_sources, 
                                modalities=opts.modalities, size=opts.size,
                                split=opts.split)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, 
                                               shuffle=True, num_workers=8, 
                                               persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, 
                                             num_workers=8, persistent_workers=True)

    return train_loader, test_loader


fc1, fc2 = None, None

def load_model(opts):
    if "resnet" in opts.model:
        model = ResNet(name=opts.model, num_classes=2)
    
    elif "-hr" in opts.model:
        model = HierarchicalResidualFT(encoder=opts.model.replace('-hr', ''), pretrained=False)
        if opts.lambd > 0:
            model = HierarchicalResidualFTNorm(encoder=opts.model.replace('-hr', ''), pretrained=False)

            if opts.fc_reg:
                model = HierarchicalResidualFTNorm2(encoder=opts.model.replace('-hr', ''), pretrained=False)
    
    if opts.freeze_encoder:
        print("Freezeing encoder")
        for p in model.encoder.parameters():
            p.requires_grad = False
        
        if opts.lambd > 0 or opts.finetune_conv:
            if "-hr" in opts.model:
                for p in model.encoder[0][-2].parameters():
                    p.requires_grad = True
    
    model = model.to(opts.device)

    if opts.load_weights is not None:
        weights = torch.load(opts.load_weights, map_location=opts.device)
        if isinstance(weights['model'], torch.nn.Module):
            model = weights['model']
        else:
            model.load_state_dict(weights['model'])


    if "-hr" in opts.model:
        global fc1
        global fc2
        
        model.fc1 = torch.nn.Linear(model.num_ft, 128).to(opts.device)
        model.fc2 = torch.nn.Linear(128, 2).to(opts.device)

        if opts.same_init and fc1 is None:
            fc1 = copy.deepcopy(model.fc1)
            fc2 = copy.deepcopy(model.fc2)

        elif opts.same_init and fc1 is not None:
            model.fc1 = copy.deepcopy(fc1)
            model.fc2 = copy.deepcopy(fc2)         
    
    cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.56, 0.43])).to(opts.device)
    def criterion(outputs, labels, feats, bias_labels):
        return opts.alpha * cross_entropy(outputs, labels) + opts.lambd*fairkl(feats, labels, bias_labels)
    return model, criterion

def load_optimizer(model, opts):
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=opts.lr, weight_decay=opts.weight_decay)

    if opts.lr_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs, 
                                                               verbose=True)
    elif opts.lr_decay == 'step':
        milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, verbose=True)
    
    elif opts.lr_decay == 'none':
        scheduler = None
    
    print(f"{len(list(parameters))} to optimize")
    
    return optimizer, scheduler

def train(train_loader, model, criterion, optimizer, opts, epoch, scaler=None):
    loss = AverageMeter()
    accuracy = BalancedAccuracy()
    auc = RocAuc()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()


    t1 = time.time()
    for idx, (images, covid, rx, age, institution) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images = images.to(opts.device)
        covid = covid.to(opts.device)
        institution = institution.to(opts.device)

        bsz = images.shape[0]

        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast(scaler is not None):
                logits, feats = model(images)
                running_loss = criterion(logits, covid, feats, institution) 
        
        optimizer.zero_grad()
        if scaler is None:
            running_loss.backward()
            optimizer.step()
        else:
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss.update(running_loss.item(), bsz)
        acc = accuracy(logits, covid)
        auc_score = auc(logits, covid) 

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"CE {loss.avg:.3f}\t"
                  f"acc@1 {acc:.3f}\t"
                  f"auc {auc_score:.3f}")

    return loss.avg, accuracy.res, auc.res, batch_time.avg, data_time.avg

@torch.no_grad()
def test(test_loader, model, criterion, opts, epoch):
    loss = AverageMeter()
    accuracy = BalancedAccuracy()
    auc = RocAuc()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    outputs = []
    labels = []
    institutions = []

    model.eval()

    t1 = time.time()
    for idx, (images, covid, rx, age, institution) in enumerate(test_loader):
        data_time.update(time.time() - t1)
        
        images = images.to(opts.device)
        covid = covid.to(opts.device)
        institution = institution.to(opts.device)
        bsz = images.shape[0]

        logits, feats = model(images)
        running_loss = criterion(logits, covid, feats, institution)
        
        outputs.append(logits.detach())
        labels.append(covid.detach())
        institutions.append(institution)

        loss.update(running_loss.item(), bsz)
        acc = accuracy(logits, covid)
        auc_score = auc(logits, covid) 

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(test_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Test: [{epoch}][{idx + 1}/{len(test_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"CE {loss.avg:.3f}\t"
                  f"acc@1 {acc:.3f}\t"
                  f"auc {auc_score:.3f}")

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    institutions = torch.cat(institutions, dim=0)

    probs = torch.softmax(outputs, dim=1)
    test_loader.dataset.df['prob0'] = probs[:, 0].cpu().numpy()
    test_loader.dataset.df['prob1'] = probs[:, 1].cpu().numpy()
    test_loader.dataset.df['pred'] = outputs.argmax(dim=1).cpu().numpy()
    test_loader.dataset.df.to_csv(os.path.join(opts.output_dir, f"test_preds.csv"))

    institution_metrics = []
    for institution in torch.unique(institutions).sort()[0]:
        idx = institutions == institution
        curr_outputs = outputs[idx]
        curr_labels = labels[idx]

        inst_acc = BalancedAccuracy()(curr_outputs, curr_labels)

        try:
            inst_auc = RocAuc()(curr_outputs, curr_labels)
        except:
            inst_auc = 0.
        
        institution_metrics.append({
            'acc@1': inst_acc,
            'auc': inst_auc 
        })


    return loss.avg, accuracy.res, auc.res, institution_metrics

def save_checkpoint(path, model, optimizer, scheduler, opts, epoch):
    torch.save({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'opts': opts,
        'epoch': epoch
    }, path)


def main(opts, group=None):
    checkpoint = None
    start_epoch = 1
    if opts.restore is not None:
        checkpoint = torch.load(opts.restore, map_location=opts.device)
        opts = checkpoint['opts']
        start_epoch = checkpoint['epoch']

    set_seed(opts.trial)

    train_loader, test_loader = load_data(opts)
    model, criterion = load_model(opts)
    optimizer, scheduler = load_optimizer(model, opts)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler']['state_dict'])
    
    model_name = opts.model
    if opts.fc_reg:
        model_name = f"{opts.model}_fc_reg"
        
    run_name = (f"{opts.prefix}corsa_{''.join(opts.modalities)}_{'x'.join(map(str, opts.size))}_"
                f"{''.join(map(str, opts.train_sources))}_{''.join(map(str, opts.test_sources))}_"
                f"aug_{'_'.join(opts.aug)}_"
                f"split{opts.split}_"
                f"{model_name}_{opts.optimizer}_bsz{opts.batch_size}_"
                f"lr{opts.lr}_{opts.lr_decay}_"
                f"ftconv{opts.finetune_conv}_"
                f"alpha{opts.alpha}_lambd{opts.lambd}_"
                f"trial{opts.trial}_same_init{opts.same_init}")
    tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)
    save_dir = os.path.join(opts.save_dir, f"corsa_models", run_name)

    ensure_dir(tb_dir)
    ensure_dir(save_dir)

    opts.model_class = model.__class__.__name__
    opts.optimizer_class = optimizer.__class__.__name__
    opts.scheduler = scheduler.__class__.__name__ if scheduler is not None else None
    opts.torch_version = torch.__version__
    opts.torchvision_version = torchvision.__version__
    opts.output_dir = save_dir

    wandb.init(project="corsa", entity="eidos", config=opts, name=run_name, group=group, sync_tensorboard=True)
    print('Config:', opts)
    print('Model:', model)
    print('Optimizer:', optimizer)
    print('Scheduler:', scheduler)
    print('torch version:', opts.torch_version)
    print('torchvision version:', opts.torchvision_version)
    print('Run id:', wandb.run.id)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.
    for epoch in range(start_epoch, opts.epochs + 1):
        t1 = time.time()
        loss_train, accuracy_train, auc_train, batch_time, data_time \
            = train(train_loader, model, criterion, optimizer, opts, epoch, scaler)
        t2 = time.time()

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("train/loss", loss_train, epoch)
        writer.add_scalar("train/acc@1", accuracy_train, epoch)
        writer.add_scalar("train/auc", auc_train, epoch)

        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} "
              f"acc {accuracy_train:.2f} auc {auc_train:.2f} loss {loss_train:.4f}")
        
        if scheduler is not None:
            scheduler.step()

        if (epoch % opts.test_freq == 0) or epoch == 1 or epoch == opts.epochs:
            loss_test, accuracy_test, auc_test, institution_metrics \
                = test(test_loader, model, criterion, opts, epoch)
            writer.add_scalar("test/loss", loss_test, epoch)
            writer.add_scalar("test/acc@1", accuracy_test, epoch)
            writer.add_scalar("test/auc", auc_test, epoch)
            print(f"test accuracy {accuracy_test:.2f} test auc {auc_test:.2f}")

            for idx, metrics in enumerate(institution_metrics):
                writer.add_scalar(f"test/institution{idx}/acc@1", metrics["acc@1"], epoch)
                writer.add_scalar(f"test/institution{idx}/auc", metrics["auc"], epoch)
                print(f"Institution {idx} ({corda.id2institution[idx]}) acc@1 {metrics['acc@1']:.3f} auc {metrics['auc']:.3f}")
            print()

            if accuracy_test > best_acc:
                best_acc = accuracy_test
                save_checkpoint(os.path.join(save_dir, "best.pth"), model, optimizer, scheduler, opts, epoch)
    
        writer.add_scalar("best_acc@1", best_acc, epoch)
        save_checkpoint(os.path.join(save_dir, "weights.pth"), model, optimizer, scheduler, opts, epoch)

    print(f"best accuracy: {best_acc:.2f}")
    wandb.finish()

    return loss_test, accuracy_test, auc_test, institution_metrics

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

if __name__ == '__main__':
    opts = parse_arguments()

    if not opts.cv:
        main(opts)
    else:
        fold_loss, fold_acc, fold_auc, fold_metrics = [], [], [], []

        group_id = "cv-" + str(uuid.uuid4())
        for i in range(4):
            opts.split = i
            #p1 = Process(target=main, args=(opts, group_id))
            #p1.start()
            #p1.join()

            loss, acc, auc, metrics = main(opts, group=group_id)
            fold_loss.append(loss)
            fold_acc.append(acc)
            fold_auc.append(auc)
            fold_metrics.append(metrics)


        print("\n\n ------------- CV FINISHED -------------")
        print(f"Loss.....: {np.mean(fold_loss):.4f} ± {np.std(fold_loss):.4f}")
        print(f"BAcc@1...: {np.mean(fold_acc)*100:.2f}% ± {np.std(fold_acc)*100:.2f}")
        print(f"AUC......: {np.mean(fold_auc[:-1]):.2f} ± {np.std(fold_auc[:-1])*100:.2f}\n")
        
        for institution in range(4):
            bacc = [m[institution]["acc@1"] for m in fold_metrics]
            auc = [m[institution]["auc"] for m in fold_metrics]
            print(f"Institution {institution} ({corda.id2institution[institution]}):\t"
                  f"Bacc@1 {np.mean(bacc)*100:.2f}% ± {np.std(bacc)*100:.2f}\t"
                  f"AUC {np.mean(auc):.2f} ± {np.std(auc):.2f}")