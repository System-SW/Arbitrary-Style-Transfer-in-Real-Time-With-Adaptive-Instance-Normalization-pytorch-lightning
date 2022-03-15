from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from models import AdaptiveInstanceNormalization as AdaIN
from models import Net


def parsing_batch(
    batch_data: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    content = batch_data["content"]
    style = batch_data["style"]
    return content, style


class TestModels:
    @pytest.fixture(scope="class", params=[True, False])
    def features(
        self,
        request,
        batch: Tuple[torch.Tensor, torch.Tensor],
        encoder: nn.Module,
    ):
        encoder = encoder if request.param else encoder.eval()
        content, style = parsing_batch(batch)
        return encoder(content), encoder(style)

    @pytest.fixture(scope="class")
    def adain(self):
        return AdaIN()

    @pytest.mark.parametrize("train", [True, False])
    def test_encoder(self, encoder: nn.Module, batch, train):
        encoder = encoder if train else encoder.eval()
        content, style = parsing_batch(batch)
        x_c = encoder(content)
        x_s = encoder(style)
        assert x_c.shape == x_s.shape

    @pytest.mark.parametrize("train", [True, False])
    def test_decoder(self, args, features, train, decoder: nn.Module):
        decoder = decoder if train else decoder.eval()
        content, _ = features
        styled = decoder(content)
        assert list(styled.shape) == [
            args.batch_size,
            3,
            args.image_size,
            args.image_size,
        ]

    @pytest.mark.parametrize("train", [True, False])
    def test_adain(self, batch, train, adain: AdaIN):
        adain = adain if train else adain.eval()
        content, style = parsing_batch(batch)

        c_mean, c_std = adain.calc_mean_std(content)
        s_mean, s_std = adain.calc_mean_std(style)

        assert c_mean.shape == s_mean.shape
        assert c_std.shape == s_std.shape

        styled_content = adain(content, style)

        assert styled_content.shape == content.shape


class TestTrainer:
    @pytest.fixture(scope="class")
    def model(self, args):
        return Net(args)

    @pytest.fixture(scope="class")
    def test(self, args):
        return Net(args)
