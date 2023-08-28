import argparse
import os
import random
import shutil
import time
import numpy as np
import pprint
import math

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from datasets.cifar10 import IMBALANCECIFAR10
from datasets.cifar100 import IMBALANCECIFAR100
from datasets.tiny_imagenet import IMBALANETINYIMGNET
from datasets.ictext import IMBALANCEICTEXT

from models import resnet
from models import resnet_cifar

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy
from utils.transforms import SquarePad

from losses.ffds import FFDS
import torchvision.transforms as TF
from collections import OrderedDict

import wandb


def setup_net(model, num_classes):
    if getattr(model, "fc", None):
        children = list(model.children())
        classifier = nn.Sequential(
            OrderedDict([("fc1", nn.Linear(children[-1].in_features,
                                           num_classes))])
        )
        model.fc = classifier
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="FFDS training")
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0


def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info("\n" + pprint.pformat(args))
    logger.info("\n" + str(config))

    if config.wandb.project:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.name,
            resume=config.wandb.resume,
            config=config,
        )

    if config.seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(config.seed)
        np.random.seed(config.seed)
        os.environ["PYTHONHASHSEED"] = str(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    main_worker(config, logger, model_dir)


def main_worker(config, logger, model_dir):
    global best_acc1
    device = "cpu" if config.gpu is None else f"cuda:{config.gpu}"

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.dataset == "cifar10" or config.dataset == "cifar100":
        model = getattr(resnet_cifar, config.backbone)(
            num_classes=config.num_classes)

    elif config.dataset == "tiny_imagenet" or config.dataset == "ictext":
        model = getattr(resnet, config.backbone)(pretrained=config.pretrained)
        model = setup_net(model, config.num_classes)

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        logger.info("using CPU, this will be slow")

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))

            checkpoint = torch.load(config.resume, map_location=device)
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    # Data loading code
    dataset_fac = {
        "cifar10": IMBALANCECIFAR10,
        "cifar100": IMBALANCECIFAR100,
        "tiny_imagenet": IMBALANETINYIMGNET,
        "ictext": IMBALANCEICTEXT,
    }

    if config.dataset == "cifar10" or config.dataset == "cifar100":
        img_size = 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_train = TF.Compose(
            [
                TF.RandomCrop(size=img_size, padding=4),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
                TF.Normalize(mean=mean, std=std),
            ]
        )

        transform_val = TF.Compose([TF.ToTensor(), TF.Normalize(mean=mean,
                                                                std=std)])

    elif config.dataset == "tiny_imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = TF.Compose(
            [
                TF.RandomResizedCrop(size=224),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
                TF.Normalize(mean=mean, std=std),
            ]
        )

        transform_val = TF.Compose(
            [
                TF.Resize(size=256),
                TF.CenterCrop(size=224),
                TF.ToTensor(),
                TF.Normalize(mean=mean, std=std),
            ]
        )

    elif config.dataset == "ictext":
        size = (32, 32)
        mean = [0.276, 0.262, 0.261]
        std = [0.239, 0.229, 0.227]
        transform_train = TF.Compose(
            [
                SquarePad(),
                TF.Resize(size=size),
                TF.RandomHorizontalFlip(),
                TF.RandomVerticalFlip(),
                TF.ToTensor(),
                TF.Normalize(mean=mean, std=std),
            ]
        )

        transform_val = TF.Compose(
            [
                SquarePad(),
                TF.Resize(size=size),
                TF.ToTensor(),
                TF.Normalize(mean=mean, std=std),
            ]
        )

    train_dataset = dataset_fac[config.dataset](
        root=config.data_path, imb_factor=config.imb_factor,
        transform=transform_train
    )
    val_dataset = dataset_fac[config.dataset](
        root=config.data_path, imb_factor=1, train=False,
        transform=transform_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )

    train_freq = torch.tensor(train_dataset.get_cls_num_list()).to(device)
    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}],
        config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        adjust_learning_rate(optimizer, epoch, config)

        if epoch == config.trigger_epoch:
            effective_num = 1.0 - torch.pow(config.beta, train_freq).to(device)
            class_weight = (1.0 - config.beta) / effective_num
            class_weight /= torch.sum(class_weight) * len(class_weight)
            criterion = FFDS(
                class_freq=train_freq,
                groups=config.groups,
                class_weight=class_weight,
                smoothing_alpha=config.smoothing_alpha,
                freq_gamma_min=config.freq_gamma_min,
                freq_gamma_max=config.freq_gamma_max,
                prob_smooth_percentage_alpha=config.prob_smooth_percentage_alpha,
                gamma_type=config.gamma_type,
            ).to(device)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config, logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, config, logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        logger.info("Best Prec@1: %.2f%%" % (best_acc1))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": round(best_acc1.item(), 2),
            },
            is_best,
            model_dir,
        )


def train(train_loader, model, criterion, optimizer, epoch, config, logger):
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    losses = AverageMeter("Loss", ":.2f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    criterion.train()

    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        if i > end_steps:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        if config.gpu is not None:
            images = images.cuda(config.gpu, non_blocking=True)
            targets = targets.cuda(config.gpu, non_blocking=True)

        output = model(images)

        loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)
    if config.wandb.project:
        wandb.log(
            {
                "epoch": epoch,
                "train/Acc@1": top1.avg,
                "train/Acc@5": top5.avg,
                "train/loss": losses.avg,
            }
        )

    if getattr(criterion, "next_epoch", None):
        criterion.next_epoch()


def validate(val_loader, model, criterion, epoch, config, logger):
    batch_time = AverageMeter("Time", ":6.2f")
    losses = AverageMeter("Loss", ":.2f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Eval: "
    )

    # switch to evaluate mode
    model.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
                targets = targets.cuda(config.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(targets, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct += (target_one_hot + predict_one_hot == 2).sum(
                dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, targets.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)

        acc_classes = correct / class_num
        head_acc = (
            acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].
            mean() * 100
        )

        med_acc = (
            acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].
            mean() * 100
        )
        tail_acc = (
            acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].
            mean() * 100
        )
        logger.info(
            "* Acc@1 {top1.avg:.2f}% Acc@5 {top5.avg:.2f}% HAcc \
                {head_acc:.2f}% MAcc {med_acc:.2f}% TAcc {tail_acc:.2f}%.".format(
                top1=top1,
                top5=top5,
                head_acc=head_acc,
                med_acc=med_acc,
                tail_acc=tail_acc,
            )
        )
        if config.wandb.project:
            wandb.log(
                {
                    "epoch": epoch,
                    "val/Acc@1": top1.avg,
                    "val/Acc@5": top5.avg,
                    "val/loss": losses.avg,
                    "val/HAcc": head_acc,
                    "val/MAcc": med_acc,
                    "val/TAcc": tail_acc,
                }
            )

    return top1.avg


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + "/current.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + "/model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    lr = config.lr
    if config.cos:
        lr_min = 0
        lr_max = config.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(epoch / config.num_epochs * 3.1415926535)
        )
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = config.lr * epoch / 5
        if config.dataset in ["cifar10", "cifar100"]:
            if epoch > 180:
                lr = config.lr * 0.01
            elif epoch > 160:
                lr = config.lr * 0.1
        elif config.dataset == "ictext":
            if epoch > 80:
                lr = config.lr * 0.0001
            elif epoch > 60:
                lr = config.lr * 0.01
        elif config.dataset == "tiny_imagenet":
            if epoch > 15:
                lr = config.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
