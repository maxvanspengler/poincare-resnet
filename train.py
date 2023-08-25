import argparse
import json
import os
import time
from datetime import datetime

import torch
from timm import utils
from torch.utils.tensorboard import SummaryWriter

from cifar10.dataloader import Cifar10DataLoaderFactory
from cifar100.dataloader import Cifar100DataLoaderFactory
from models.optimizers import initialize_optimizer
from models.resnets import parse_model_from_name

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 training")

parser.add_argument("model", type=str, help="Model name")
parser.add_argument(
    "dataset",
    type=str,
    choices=["cifar10", "cifar100"],
    help="Dataset (cifar10, cifar100)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    help="Overwrite batch size (default: 128)",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=500,
    help="Number of epochs (default: 500)",
)
parser.add_argument(
    "--opt",
    type=str,
    default="sgd",
    help="Optimizer (default: sgd)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Learning rate (default: 0.001)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-4,
    help="Weight decay (default: 1e-4)",
)
parser.add_argument(
    "-s",
    "--save",
    action="store_const",
    const=True,
    default=False,
    help="Save the model weights after training",
)
parser.add_argument(
    "--criterion",
    type=str,
    default="top1",
    choices=["losses", "top1", "top5"],
    help="Choose the metric which will determine the best result (default: top1)",
)


def main():
    args = parser.parse_args()

    # Create some strings for file management
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, "runs", args.dataset, args.model, now)
    os.makedirs(exp_dir)

    # Grab some dataset specific stuff
    if args.dataset == "cifar10":
        dataset_factory = Cifar10DataLoaderFactory
        classes = 10
    elif args.dataset == "cifar100":
        dataset_factory = Cifar100DataLoaderFactory
        classes = 100

    # Create dataloaders
    train_loader, test_loader = dataset_factory.create_train_loaders(
        batch_size=args.batch_size
    )

    # Create model
    model = parse_model_from_name(args.model, classes).cuda()

    # Initialize tensorboard logger
    writer = SummaryWriter(exp_dir)

    # Create optimizers
    optimizer = initialize_optimizer(
        model=model,
        args=args,
    )

    print(f"Using optimizer: {optimizer}")

    loss_fn = torch.nn.CrossEntropyLoss()

    best_avg_metrics = {}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()

        for idx, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = {
            "losses": utils.AverageMeter(),
            "top1": utils.AverageMeter(),
            "top5": utils.AverageMeter(),
        }

        model.eval()

        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.cuda(), target.cuda()
                output = model(input)

                loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                metrics["losses"].update(loss.data.item(), input.size(0))
                metrics["top1"].update(acc1.item(), output.size(0))
                metrics["top5"].update(acc5.item(), output.size(0))

        writer.add_scalar(f"{args.criterion}/test", metrics[args.criterion].avg, epoch)

        if (
            not best_avg_metrics
            or metrics[args.criterion].avg > best_avg_metrics[args.criterion]
        ):
            best_avg_metrics = {k: metrics[k].avg for k in metrics}
            best_model_state = model.state_dict()

        print(
            f"Epoch {epoch}:  "
            f"Time: {time.time() - epoch_start:.3f}  "
            f"Loss: {metrics['losses'].avg:>7.4f}  "
            f"Acc@1: {metrics['top1'].avg:>7.4f}  "
            f"Acc@5: {metrics['top5'].avg:>7.4f}"
        )

    output_dict = {
        "best_model_state": best_model_state,
        "best_avg_metrics": best_avg_metrics,
        "last_model_state": model.state_dict(),
        "last_avg_metrics": {k: metrics[k].avg for k in metrics},
    }

    # Store model weights
    if args.save:
        torch.save(
            output_dict["last_model_state"],
            os.path.join(exp_dir, f"{args.model}_weights.pth"),
        )

        weights_dir = os.path.join(dir_path, "weights", args.dataset, args.model)
        os.makedirs(weights_dir)
        torch.save(
            output_dict["last_model_state"],
            os.path.join(weights_dir, f"{args.model}_weights.pth"),
        )

    # Store metrics
    with open(f"{exp_dir}/metrics.json", "w") as file:
        json.dump(output_dict["last_avg_metrics"], file, indent=4)


if __name__ == "__main__":
    main()
