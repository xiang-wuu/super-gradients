from copy import deepcopy

import torch
from torch.onnx import TrainingMode
import torch.nn as nn
import onnxsim
import onnx
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

try:
    from pytorch_quantization import nn as quant_nn

    _imported_pytorch_quantization_failure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warning("Failed to import pytorch_quantization")
    _imported_pytorch_quantization_failure = import_err


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[0]
        scores, classes = torch.max(x[1], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def export_quantized_module_to_onnx(
    model: torch.nn.Module, onnx_filename: str, input_shape: tuple, train: bool = False, to_cpu: bool = True, deepcopy_model=False, **kwargs
):
    """
    Method for exporting onnx after QAT.

    :param to_cpu: transfer model to CPU before converting to ONNX, dirty workaround when model's tensors are on different devices
    :param train: export model in training mode
    :param model: torch.nn.Module, model to export
    :param onnx_filename: str, target path for the onnx file,
    :param input_shape: tuple, input shape (usually BCHW)
    :param deepcopy_model: Whether to export deepcopy(model). Necessary in case further training is performed and
     prep_model_for_conversion makes the network un-trainable (i.e RepVGG blocks).
    """
    if _imported_pytorch_quantization_failure is not None:
        raise _imported_pytorch_quantization_failure

    if deepcopy_model:
        model = deepcopy(model)

    use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    logger.info("Creating ONNX file: " + onnx_filename)

    if train:
        training_mode = TrainingMode.TRAINING
        model.train()
    else:
        training_mode = TrainingMode.EVAL
        model.eval()
        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**kwargs)

    # workaround when model.prep_model_for_conversion does reparametrization
    # and tensors get scattered to different devices
    if to_cpu:
        export_model = model.cpu()
    else:
        export_model = model

    export_model = nn.Sequential(export_model, DeepStreamOutput())

    dynamic_axes = {"input": {0: "batch"}, "boxes": {0: "batch"}, "scores": {0: "batch"}, "classes": {0: "batch"}}

    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_filename,
        verbose=False,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "scores", "classes"],
        dynamic_axes=dynamic_axes,
        training=training_mode,
    )

    model_onnx = onnx.load(onnx_filename)
    model_onnx, _ = onnxsim.simplify(model_onnx)
    onnx.save(model_onnx, onnx_filename)

    # Restore functions of quant_nn back as expected
    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state
