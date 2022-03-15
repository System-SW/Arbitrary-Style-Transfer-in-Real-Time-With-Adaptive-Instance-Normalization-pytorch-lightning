from models import Net
import torch
import torch.nn as nn
import pytest
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

    def test_calc_loss(self, train_output, model):
        g_t, t, style_feats = train_output
        loss_c, loss_s = model.calc_loss(style_feats, t, g_t)
        assert list(loss_c.shape) == []
        assert list(loss_s.shape) == []
