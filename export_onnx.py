from torch import nn
from DAM4SAM.dam4sam_tracker import DAM4SAMTracker
import torch
import onnx
import os

tracker = DAM4SAMTracker("sam21pp-L")
assert tracker.predictor.device != "cpu", "bad device"
device = tracker.predictor.device
save_path = "model"
os.makedirs(save_path, exist_ok=True)


def export_tracker_predictor_image_encoder(onnx_name="image_encoder.onnx"):
    module = tracker.predictor.image_encoder
    ori_forward = module.forward
    module.forward = module.inference

    dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
    torch.onnx.export(
        module,
        dummy_input,
        os.path.join(save_path, onnx_name),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["feature0", "feature1", "feature2", "pos0", "pos1", "pos2"],
        dynamic_axes={
            "image": {0: "N"},
            "feature0": {0: "N"},
            "feature1": {0: "N"},
            "feature2": {0: "N"},
            "pos0": {0: "N"},
            "pos1": {0: "N"},
            "pos2": {0: "N"},
        },
    )

    module.forward = ori_forward


def export_tracker_predictor_forward_image(
    onnx_name="forward_image.onnx", simplify_onnx=True
):
    module = tracker.predictor
    ori_forward = module.forward
    module.forward = module.forward_image_inference

    dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
    torch.onnx.export(
        module,
        dummy_input,
        os.path.join(save_path, onnx_name),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=[
            "vision_features",
            "feature0",
            "feature1",
            "feature2",
            "pos0",
            "pos1",
            "pos2",
        ],
        dynamic_axes={
            "image": {0: "N"},
            "vision_features": {0: "N"},
            "feature0": {0: "N"},
            "feature1": {0: "N"},
            "feature2": {0: "N"},
            "pos0": {0: "N"},
            "pos1": {0: "N"},
            "pos2": {0: "N"},
        },
    )

    from onnxsim import simplify

    onnx_model = onnx.load(onnx_name)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_name.replace(".onnx", "_opt.onnx"))
    module.forward = ori_forward


export_tracker_predictor_forward_image()

# dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
# torch.onnx.export(
#     module,
#     dummy_input,
#     "model/deepsort_embedding.onnx",
#     export_params=True,
#     opset_version=14,
#     do_constant_folding=True,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
# )
