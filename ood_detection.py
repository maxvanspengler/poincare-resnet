import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn

import ood_utils.svhn_loader as svhn
from cifar10.dataloader import Cifar10DataLoaderFactory
from cifar100.dataloader import Cifar100DataLoaderFactory
from config import config
from models.resnets import parse_model_from_name
from ood_utils.display_results import (
    get_measures,
    print_measures,
    print_measures_with_std,
    show_performance,
)

parser = argparse.ArgumentParser(
    description="Evaluates a CIFAR OOD Detector",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", type=str, default="allconv", help="Choose architecture.")
parser.add_argument("--dataset", type=str, default="cifar10", help="Choose dataset.")
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument(
    "--num_to_avg", type=int, default=1, help="Average measures across num_to_avg runs."
)
parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
parser.add_argument("--prefetch", type=int, default=2, help="Pre-fetching threads.")
parser.add_argument("--T", default=1.0, type=float, help="temperature")
args = parser.parse_args()
print(args)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

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

# Detection Prelims
ood_num_examples = len(test_loader) * args.batch_size // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_loader))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.batch_size and in_dist is False:
                break

            data = data.cuda()

            output = model(data)
            smax = to_np(F.softmax(output, dim=1))

            _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return (
            concat(_score).copy(),
            concat(_right_score).copy(),
            concat(_wrong_score).copy(),
        )
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print("Error Rate {:.2f}".format(100 * num_wrong / (num_wrong + num_right)))
# End Detection Prelims

print("\nUsing CIFAR-10 as typical data") if classes == 10 else print(
    "\nUsing CIFAR-100 as typical data"
)

# Error Detection
print("\n\nError Detection")
show_performance(wrong_score, right_score, method_name=args.model)

# OOD Detection
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])

    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.model)
    else:
        print_measures(auroc, aupr, fpr, args.model)


# Textures
ood_data = dset.ImageFolder(
    root=config["DATASETS"]["Textures"],
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
ood_loader = torch.utils.data.DataLoader(
    ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
print("\n\nTexture Detection")
get_and_print_results(ood_loader)

# SVHN
ood_data = svhn.SVHN(
    root=config["DATASETS"]["SVHN"],
    split="test",
    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
    download=False,
)
ood_loader = torch.utils.data.DataLoader(
    ood_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
)
print("\n\nSVHN Detection")
get_and_print_results(ood_loader)

# Places365
ood_data = dset.ImageFolder(
    root=config["DATASETS"]["Places365"],
    transform=trn.Compose(
        [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]
    ),
)
ood_loader = torch.utils.data.DataLoader(
    ood_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
)
print("\n\nPlaces365 Detection")
get_and_print_results(ood_loader)

# Mean Results
print("\n\nMean Test Results!!!!!")
print_measures(
    np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.model
)
