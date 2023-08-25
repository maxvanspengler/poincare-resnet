import geoopt
import torch
import torch.nn as nn

optimizer_dict = {
    "sgd": geoopt.optim.RiemannianSGD,
    "adam": geoopt.optim.RiemannianAdam,
}


allowed_opt_kwargs = {
    "sgd": [
        "lr",
        "momentum",
        "weight_decay",
    ],
    "adam": [
        "lr",
        "weight_decay",
    ],
}


def parse_optimizer_kwargs(args: dict) -> list:
    opt = args["opt"]
    opt_kwargs = {}
    for key in args.keys():
        if key in allowed_opt_kwargs[opt]:
            opt_kwargs[key] = args[key]

    return opt, opt_kwargs


def initialize_optimizer(
    model: nn.Module,
    args,
) -> list[torch.optim.Optimizer]:
    # Parse optimizer specification and configuration
    opt, config = parse_optimizer_kwargs(vars(args))

    # Create optimizer
    optimizer = optimizer_dict[opt]
    return optimizer(model.parameters(), **config)
