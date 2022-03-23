import os

import numpy as np
import onnxruntime
import pytest
import torch
import torch.nn as nn
from models import Net

from tests.test_models import parsing_batch


class TestNet:
    @pytest.fixture(scope="class")
    def model(self, args, decoder: nn.Module, encoder: nn.Module):
        return Net(args, encoder, decoder)

    @pytest.fixture(scope="class")
    def inputs(self, batch):
        content, style = parsing_batch(batch)
        return content, style

    @pytest.fixture(scope="class")
    def train_output(self, inputs, model):
        content, style = inputs
        g_t, t, style_feats = model(content, style)
        return g_t, t, style_feats

    @pytest.fixture(scope="class")
    def example_inputs(self):
        return (
            torch.rand([1, 3, 256, 256]),
            torch.rand([1, 3, 256, 256]),
            torch.zeros(1),
        )

    def test_forward(self, inputs, train_output):
        g_t, t, style_feats = train_output
        content, style = inputs
        assert g_t.shape == content.shape

    def test_eval_forward(self, inputs, model):
        model = model.eval()
        content, style = inputs
        g_t = model(content, style)
        assert g_t.shape == content.shape
        model.train(True)

    def test_save_model(self, temp_dir_f, model):
        path = os.path.join(temp_dir_f.name, "model.pth")
        state_dict = model.state_dict()
        torch.save(state_dict, path)
        assert os.path.exists(path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    def test_save_torchscript(self, temp_dir_f, example_inputs, model):
        path = os.path.join(temp_dir_f.name, "model.pt.zip")
        model.to_torchscript(path, "trace", example_inputs=example_inputs)
        assert os.path.exists(path)

    def test_save_onnx(self, args, temp_dir_f, example_inputs, model):
        path = os.path.join(temp_dir_f.name, "model.onnx")
        input_names = ["content", "style", "alpha"]
        output_names = ["output"]
        dynamic_axes = {
            input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
            input_names[1]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
            output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        }
        model.to_onnx(
            file_path=path,
            input_sample=example_inputs,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.onnx_opset_version,
            dynamic_axes=dynamic_axes,
        )
        assert os.path.exists(path)

    def test_load_torchscript(self, temp_dir_f, example_inputs, model):
        path = os.path.join(temp_dir_f.name, "model.pt.zip")
        model = torch.jit.load(path)
        content, style, _ = example_inputs

        for alpha in torch.Tensor(np.arange(0.0, 0.3, 0.1, dtype=np.float32)):
            g_t = model(content, style, alpha)
            assert g_t.shape == content.shape

    @pytest.mark.parametrize("size", [(128, 256), (512, 256)])
    def test_load_onnx(self, temp_dir_f, example_inputs, size):
        path = os.path.join(temp_dir_f.name, "model.onnx")
        session = onnxruntime.InferenceSession(path)
        content, style, _ = example_inputs
        content = torch.zeros([1, 3, *size]).numpy()
        style = style.numpy()
        inputs_tag = session.get_inputs()
        outputs_tag = session.get_outputs()
        for alpha in np.arange(0.0, 0.3, 0.1, dtype=np.float32):
            inputs = {
                inputs_tag[0].name: content,
                inputs_tag[1].name: style,
                inputs_tag[2].name: [alpha],
            }
            g_t = session.run([outputs_tag[0].name], inputs)[0]
            assert list(g_t.shape) == list(content.shape)
