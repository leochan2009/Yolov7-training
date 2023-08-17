import torch
from pytorch_accelerated.utils import local_process_zero_first
from yolov7.models.model_configs import (
    get_yolov7_config,
    get_yolov7_d6_config,
    get_yolov7_e6_config,
    get_yolov7_e6e_config,
    get_yolov7_tiny_config,
    get_yolov7_w6_config,
    get_yolov7x_config,
)
from yolov7.models.yolo import Yolov7Model
from yolov7.utils import intersect_dicts

MODEL_CONFIGS = {
    "yolov7": get_yolov7_config,
    "yolov7x": get_yolov7x_config,
    "yolov7-tiny": get_yolov7_tiny_config,
    "yolov7-w6": get_yolov7_w6_config,
    "yolov7-d6": get_yolov7_d6_config,
    "yolov7-e6": get_yolov7_e6_config,
    "yolov7-e6e": get_yolov7_e6e_config,
}


@local_process_zero_first
def create_yolov7_model(
    architecture,
    num_classes=80,
    anchor_sizes_per_layer=None,
    num_channels=3,
    pretrained=True,
    pretrainedWeights=""
):
    config = MODEL_CONFIGS[architecture](
        num_classes=num_classes,
        anchor_sizes_per_layer=anchor_sizes_per_layer,
        num_channels=num_channels,
    )
    model = Yolov7Model(model_config=config)
    if pretrainedWeights == "":
        state_dict = intersect_dicts(
            torch.hub.load_state_dict_from_url(config["state_dict_path"], progress=False),
            model.state_dict(),
            exclude=["anchor"],
        )
        pretrainedWeights = config["state_dict_path"]
    else:
        checkpoint = torch.load(pretrainedWeights, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        state_dict = checkpoint['model_state_dict']
    if pretrained:
        if state_dict is None:
            raise ValueError(
                "Pretrained weights are not available for this architecture"
            )
        try:
            # load state dict
            model.load_state_dict(state_dict, strict=False)
            print(
                f"Transferred {len(state_dict)}/{len(model.state_dict())} items from {pretrainedWeights}"
            )
        except Exception as e:
            print(f"Unable to load pretrained model weights from {pretrainedWeights}")
            print(e)
    return model
