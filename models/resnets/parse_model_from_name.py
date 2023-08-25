from .euclidean import EuclideanResNet
from .euclidean_w_hyp_class import EuclideanResNetWHypClass
from .hyperbolic import HyperbolicResNet


def parse_model_from_name(
    model_name: str,
    classes: int,
) -> EuclideanResNet | EuclideanResNetWHypClass | HyperbolicResNet:
    keys = model_name.split("-")
    model_type = keys[0]
    channel_dims = [int(k) for k in keys[1:4]]
    # depths explanation: 2 layers outside groups (-2), 2 * 3 = 6 layers added
    # when a residual block (2 layers) is added to each group (*3)
    depths = 3 * [(int(keys[-1]) - 2) // 6]

    if model_type == "euclidean":
        model_class = EuclideanResNet
    elif model_type == "euclideanwhypclass":
        model_class = EuclideanResNetWHypClass
    elif model_type == "hyperbolic":
        model_class = HyperbolicResNet

    return model_class(
        classes=classes,
        channel_dims=channel_dims,
        depths=depths,
    )
