import torchvision.transforms as transforms


def get_standard_transform(train: bool = True) -> transforms.Compose:
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    if train:
        transform.extend(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    return transforms.Compose(transform)
