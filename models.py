import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid

import wandb


class Net(pl.LightningModule):
    def __init__(self, args, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.save_hyperparameters(args)
        self.automatic_optimization = False

        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.sample_image = None
        self.AdaIN = AdaptiveInstanceNormalization()

        # fix the encoder
        for name in ["enc_1", "enc_2", "enc_3", "enc_4"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer

    def lr_lambda(self, optimizer, step):
        lr = self.hparams.lr / (1.0 + self.hparams.lr_decay * step)
        return lr

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, "enc_{:d}".format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return F.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = self.AdaIN.calc_mean_std(input)
        target_mean, target_std = self.AdaIN.calc_mean_std(target)
        return F.mse_loss(input_mean, target_mean) + F.mse_loss(
            input_std, target_std
        )

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0
    ):
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = self.AdaIN(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        if self.training:
            return g_t, style_feats, t
        return g_t

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        self.lr_lambda(optimizer, self.global_step)
        content = batch["content"]
        style = batch["style"]

        if self.sample_image is None:
            self.sample_image = [content, style]

        g_t, style_feats, t = self(content, style)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])

        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        loss = (
            loss_c * self.hparams.content_weight
            + loss_s * self.hparams.style_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            self.log_dict(
                {
                    "loss": loss,
                    "content_loss": loss_c,
                    "style_loss": loss_s,
                },
                prog_bar=True,
            )
        if batch_idx % 100 == 0:
            self.sampling_step(self.sample_image)

    @torch.no_grad()
    def sampling_step(self, sample_input):
        self.eval()
        content, style = sample_input
        content = content[: self.hparams.max_image_count]
        style = style[: self.hparams.max_image_count]

        g_t = self(content, style, 1.0)

        images = [
            make_grid(content, 8, 2, normalize=True),
            make_grid(style, 8, 2, normalize=True),
            make_grid(g_t, 8, 2, normalize=True),
        ]

        image = make_grid(images, 1, 2)
        image = wandb.Image(image)
        wandb.log({"image/train": image})
        self.train(True)


class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x: torch.Tensor):
        b, c = x.shape[:2]
        mean = torch.mean(x, dim=[2, 3]).view([b, c, 1, 1])
        std = torch.std(x, dim=[2, 3]).view([b, c, 1, 1]) + self.eps
        return mean, std

    def forward(
        self, content: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor:
        size = content.size()
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)

        normalized_feat = (
            content - content_mean.expand(size)
        ) / content_std.expand(size)
        normalized_feat = normalized_feat * style_std.expand(
            size
        ) + style_mean.expand(size)
        return normalized_feat


def Decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def Encoder():
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-4
    )
