import argparse
import os
import random
import time
import numpy as np
import pprint

import torch
import torch.nn as nn
import torch.nn.parallel
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
    parser = argparse.ArgumentParser(description="FFDS evaluation")
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

    if config.seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(config.seed)
        np.random.seed(config.seed)
        os.environ["PYTHONHASHSEED"] = str(config.seed)
        torch.manual_seed(config.seed)
        # torch.cuda.manual_seed(config.seed)
        # torch.cuda.manual_seed_all(config.seed)

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
        model = getattr(resnet, config.backbone)()
        model = setup_net(model, config.num_classes)

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        logger.info("using CPU, this will be slow")

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
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform_val = TF.Compose([TF.ToTensor(), TF.Normalize(mean=mean,
                                                                std=std)])

    elif config.dataset == "tiny_imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

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
        transform_val = TF.Compose(
            [
                SquarePad(),
                TF.Resize(size=size),
                TF.ToTensor(),
                TF.Normalize(mean=mean, std=std),
            ]
        )

    val_dataset = dataset_fac[config.dataset](
        root=config.data_path, imb_factor=1, train=False,
        transform=transform_val
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # evaluate on validation set
    validate(val_loader, model, criterion, checkpoint["epoch"], config, logger)


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
    class_num = torch.zeros(config.num_classes)
    correct = torch.zeros(config.num_classes)

    if config.gpu is not None:
        class_num = class_num.cuda(config.gpu)
        correct = correct.cuda(config.gpu)

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct += (target_one_hot + predict_one_hot == 2).sum(
                dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

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


if __name__ == "__main__":
    main()
