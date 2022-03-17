import os
import tempfile

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

    def test_save_model(self, model):
        temp_f = tempfile.TemporaryDirectory()
        path = temp_f.name
        path = os.path.join(path, "model.pt.zip")
        state_dict = model.state_dict()
        torch.save(state_dict, path)
        assert os.path.exists(path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
