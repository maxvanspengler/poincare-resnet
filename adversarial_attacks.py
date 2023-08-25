import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from timm import utils
from tqdm import tqdm

from cifar10.dataloader import Cifar10DataLoaderFactory
from cifar100.dataloader import Cifar100DataLoaderFactory
from models.resnets import parse_model_from_name

parser = argparse.ArgumentParser(description="PyTorch adversarial attack evaluation")

parser.add_argument("model", type=str, help="Model name")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("-e", "--epsilon", type=float)
parser.add_argument("-b", "--batch-size", type=int, default=128)


def create_metrics_dict():
    return {
        "losses": utils.AverageMeter(),
        "top1": utils.AverageMeter(),
        "top5": utils.AverageMeter(),
    }


def update_metrics_dict(metrics, input, loss, acc1, acc5):
    metrics["losses"].update(loss.data.item(), input.size(0))
    metrics["top1"].update(acc1.item(), input.size(0))
    metrics["top5"].update(acc5.item(), input.size(0))
    return metrics


def main() -> dict:
    args = parser.parse_args()

    if args.dataset == "cifar10":
        _, test_loader = Cifar10DataLoaderFactory.create_train_loaders(
            batch_size=args.batch_size
        )
        classes = 10
    elif args.dataset == "cifar100":
        _, test_loader = Cifar100DataLoaderFactory.create_train_loaders(
            batch_size=args.batch_size
        )
        classes = 100

    model = parse_model_from_name(model_name=args.model, classes=classes).cuda()
    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weights",
        args.dataset,
        args.model,
        f"{args.model}_weights.pth",
    )
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    metrics = {attack: create_metrics_dict() for attack in ["clean", "fgm"]}

    for input, target in tqdm(test_loader):
        input, target = input.cuda(), target.cuda()
        input_fgm = fast_gradient_method(
            model_fn=model, x=input, eps=args.epsilon, norm=np.inf
        )

        output = model(input)
        output_fgm = model(input_fgm)

        loss = loss_fn(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        metrics["clean"] = update_metrics_dict(
            metrics["clean"], input, loss, acc1, acc5
        )

        loss_fgm = loss_fn(output_fgm, target)
        acc1_fgm, acc5_fgm = utils.accuracy(output_fgm, target, topk=(1, 5))
        metrics["fgm"] = update_metrics_dict(
            metrics["fgm"], input_fgm, loss_fgm, acc1_fgm, acc5_fgm
        )

        del loss, loss_fgm

    for attack in metrics.keys():
        print(
            f"Metrics for {attack}:  "
            f"Loss: {metrics[attack]['losses'].avg:>7.4f}  "
            f"Acc@1: {metrics[attack]['top1'].avg:>7.4f}  "
            f"Acc@5: {metrics[attack]['top5'].avg:>7.4f}"
        )


if __name__ == "__main__":
    main()
