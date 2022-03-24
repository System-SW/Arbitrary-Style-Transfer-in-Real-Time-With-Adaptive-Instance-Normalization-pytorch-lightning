import os

import cv2
import numpy as np
import streamlit as st
import onnxruntime


@st.cache()
@st.experimental_singleton
class Model:
    def __init__(self, encoder_path: str, decoder_path: str) -> None:
        self.encoder = onnxruntime.InferenceSession(encoder_path)
        self.decoder = onnxruntime.InferenceSession(decoder_path)

    @classmethod
    @st.cache()
    def preprocess(cls, image: np.ndarray, size=512) -> np.ndarray:
        if isinstance(size, int):
            image = cv2.resize(image, dsize=(size, size))
        image = image.astype(np.float32) / 255
        image = image.transpose(2, 1, 0)
        image = np.expand_dims(image, axis=0)
        return image

    @st.cache
    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        inputs = {"input": image}
        feature = self.encoder.run(["output"], inputs)[0]
        return feature

    @st.cache
    def adaptive_instance_normalization(
        self, content: np.ndarray, style: np.ndarray
    ) -> np.ndarray:
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)

        normalized_content = (content - content_mean) / content_std
        normalized_content = normalized_content * style_std + style_mean
        return normalized_content

    @st.cache
    def calc_mean_std(self, x: np.ndarray):
        N, C, H, W = x.shape
        x = x.reshape([1, C, -1])
        mean = np.mean(x, axis=2).reshape([N, C, 1, 1])
        std = np.std(x, axis=2) + 0.00000001
        std = std.reshape([N, C, 1, 1])
        return mean, std

    @st.cache
    def decoding(self, t: np.ndarray) -> np.ndarray:
        inputs = {"input": t}
        outputs = self.decoder.run(["output"], inputs)[0]
        return outputs

    def inference(
        self, content: np.ndarray, style: np.ndarray, alpha: float
    ) -> np.ndarray:

        content_features = self.extract_feature(content)
        style_features = self.extract_feature(style)
        t = self.adaptive_instance_normalization(
            content_features, style_features
        )
        t = alpha * t + (1 - alpha) * content_features
        outputs = self.decoding(t)
        return outputs

    @classmethod
    @st.cache
    def postprocess(cls, image: np.ndarray):
        image = image * 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)[0]
        image = image.transpose(2, 1, 0)
        return image

    @st.cache
    def predict(
        self,
        content: np.ndarray,
        style: np.ndarray,
        alpha: float,
        content_size: int,
        style_size: int,
    ):
        w, h, _ = content.shape
        content = self.preprocess(content, content_size)
        style = self.preprocess(style, style_size)
        output = self.inference(content=content, style=style, alpha=alpha)
        image = self.postprocess(output)
        image = cv2.resize(image, (h, w))
        return image
