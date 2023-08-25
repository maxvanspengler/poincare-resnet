import argparse
import os

import torch
import torchvision
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from cifar10.dataloader import Cifar10DataLoaderFactory
from cifar100.dataloader import Cifar100DataLoaderFactory
from config import config
from models.resnets import parse_model_from_name

parser = argparse.ArgumentParser(description="PyTorch adversarial attack evaluation")

parser.add_argument("model", type=str, help="Model name")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument(
    "--count", "-c", type=int, default=100, help="Minimum number of images to input"
)
parser.add_argument(
    "--errors-only",
    "-e",
    action="store_true",
    help="Only create visualizatons for wrong predictions",
)

cifar10_classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


if __name__ == "__main__":
    args = parser.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == "cifar10":
        _, test_loader = Cifar10DataLoaderFactory.create_train_loaders(
            batch_size=args.batch_size
        )
        classes = 10
        original_test_data = torchvision.datasets.CIFAR10(
            root=config["DATASETS"]["Cifar10"],
            train=False,
            download=False,
            transform=ToTensor(),
        )
    elif args.dataset == "cifar100":
        _, test_loader = Cifar100DataLoaderFactory.create_train_loaders(
            batch_size=args.batch_size
        )
        classes = 100
        original_test_data = torchvision.datasets.CIFAR10(
            root=config["DATASETS"]["Cifar100"],
            train=False,
            download=False,
            transform=ToTensor(),
        )

    original_test_loader = DataLoader(
        dataset=original_test_data, batch_size=args.batch_size, shuffle=False
    )

    def create_model_from_name_and_prefix(model_name: str, prefix: str):
        model_name = f"{prefix}-{model_name}"
        model = parse_model_from_name(model_name=model_name, classes=classes).cuda()
        weights_path = os.path.join(
            root, "weights", args.dataset, model_name, f"{model_name}_weights.pth"
        )
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        model.eval()
        target_layers = [model.group3]
        return model, target_layers

    hyp_model, hyp_target_layers = create_model_from_name_and_prefix(
        model_name=args.model, prefix="hyperbolic"
    )
    euc_model, euc_target_layers = create_model_from_name_and_prefix(
        model_name=args.model, prefix="euclidean"
    )

    exp_dir = os.path.join(
        root,
        "gradcam",
        args.dataset,
        f"{'ERRORS-' if args.errors_only else ''}{args.model}",
    )
    os.makedirs(exp_dir, exist_ok=True)

    hyp_cam = GradCAM(model=hyp_model, target_layers=hyp_target_layers, use_cuda=True)
    euc_cam = GradCAM(model=euc_model, target_layers=euc_target_layers, use_cuda=True)

    total = 0
    hyp_total_correct = 0
    euc_total_correct = 0

    for batch_id, ((batch, targets), (originals, _)) in enumerate(
        zip(test_loader, original_test_loader)
    ):
        with torch.no_grad():
            hyp_logits = hyp_model(batch.cuda())
            hyp_preds = torch.argmax(hyp_logits, dim=-1)
            hyp_correct = hyp_preds == targets.cuda()

            euc_logits = euc_model(batch.cuda())
            euc_preds = torch.argmax(euc_logits, dim=-1)
            euc_correct = euc_preds == targets.cuda()

            total += batch.size(0)
            hyp_total_correct += hyp_correct.sum()
            euc_total_correct += euc_correct.sum()

        hyp_grayscale_cam = hyp_cam(input_tensor=batch, targets=None, aug_smooth=True)
        euc_grayscale_cam = euc_cam(input_tensor=batch, targets=None, aug_smooth=True)

        for im_id in range(args.batch_size):
            if args.errors_only and hyp_correct[im_id] and euc_correct[im_id]:
                continue
            output_img = Image.new("RGB", (32 * 3, 32))
            original_img = originals[im_id, :]

            hyp_visualization = show_cam_on_image(
                original_img.movedim([0], [2]).numpy(),
                hyp_grayscale_cam[im_id, :],
                use_rgb=True,
            )
            hyp_grad_img = Image.fromarray(hyp_visualization, "RGB")

            euc_visualization = show_cam_on_image(
                original_img.movedim([0], [2]).numpy(),
                euc_grayscale_cam[im_id, :],
                use_rgb=True,
            )
            euc_grad_img = Image.fromarray(euc_visualization, "RGB")

            original_img = to_pil_image(original_img)

            output_img.paste(original_img, (0, 0))
            output_img.paste(euc_grad_img, (32, 0))
            output_img.paste(hyp_grad_img, (64, 0))

            output_img.save(
                os.path.join(
                    exp_dir,
                    f"{batch_id * args.batch_size + im_id}_euc_vs_hyp_gradcam"
                    f"_{cifar10_classes[targets[im_id]]}_{cifar10_classes[euc_preds[im_id]]}"
                    f"_{cifar10_classes[hyp_preds[im_id]]}.jpeg",
                )
            )

        if (batch_id + 1) * args.batch_size >= args.count:
            break

    print("Hyp accuracy:", hyp_total_correct / total)
    print("Euc accuracy:", euc_total_correct / total)
