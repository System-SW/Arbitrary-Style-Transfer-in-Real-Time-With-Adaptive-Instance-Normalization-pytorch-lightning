import os

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.utils import make_grid


@st.cache(hash_funcs={torch.jit._script.RecursiveScriptModule: lambda _: None})
@st.experimental_singleton
class Model:
    def __init__(self, model_path) -> None:
        self.model = torch.jit.load(model_path)

    @classmethod
    def preprocess(cls, image: np.ndarray, size=512) -> torch.Tensor:
        image = Image.fromarray(image)
        if isinstance(size, int):
            image = F.resize(image, size=[size, size])
        image = F.to_tensor(image)
        image = torch.unsqueeze(image, dim=0)
        return image

    @torch.no_grad()
    def inferance(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        output = self.model(content, style, torch.Tensor([alpha])[0])
        return output

    @classmethod
    def postprocess(cls, image: torch.Tensor) -> np.ndarray:
        image = make_grid(image)
        image = (
            image.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        return Image.fromarray(image)

    @st.cache(
        hash_funcs={torch.jit._script.RecursiveScriptModule: lambda _: None}
    )
    def predict(
        self,
        content: np.ndarray,
        style: np.ndarray,
        alpha: float,
        content_size: int,
        style_size: int,
    ):
        h, w, _ = content.shape
        content = self.preprocess(content, content_size)
        style = self.preprocess(style, style_size)
        output = self.inferance(content=content, style=style, alpha=alpha)
        image = self.postprocess(output)
        return image.resize((w, h))
