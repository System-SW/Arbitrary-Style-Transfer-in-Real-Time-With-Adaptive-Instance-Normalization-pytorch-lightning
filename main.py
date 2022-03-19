import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import wandb

from dataset import build_dataloader
from models import Decoder, Encoder, Net


def main():
    parser = argparse.ArgumentParser()
    # project
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--project_name", type=str, default="Adaptive-Instance-Normalization"
    )

    parser.add_argument("--seed", type=int, default=9423)
    parser.add_argument("--logdir", type=str, default="experiment")

    # data
    parser.add_argument("--content_root_dir", type=str)
    parser.add_argument("--style_root_dir", type=str)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    # model
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--decoder", type=str, default="weights/decoder.pth")
    parser.add_argument(
        "--encoder", type=str, default="weights/vgg_normalised.pth"
    )

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument("--content_weight", type=float, default=1.0)

    # logger
    parser.add_argument("--upload_artifacts", action="store_true")
    parser.add_argument("--onnx_opset_version", type=int, default=11)
    parser.add_argument("--max_image_count", type=int, default=8)

    args = pl.Trainer.parse_argparser(parser.parse_args())

    training(args)


def training(args):
    assert os.path.exists(args.encoder), f"{args.encoder} is not found"
    pl.seed_everything(args.seed)

    ########### BUILD DATASET ############
    content_dataloader = build_dataloader(args, args.content_root_dir)
    style_dataloader = build_dataloader(args, args.style_root_dir)

    ############### MODEL ################
    encoder = Encoder()
    encoder.load_state_dict(torch.load(args.encoder))
    encoder = nn.Sequential(*list(encoder.children())[:31])

    decoder = Decoder()
    if os.path.exists(args.decoder):
        print("decoder load!")
        decoder.load_state_dict(torch.load(args.decoder, map_location="cpu"))

    model = Net(args, encoder=encoder, decoder=decoder)

    ############### LOGGER ################
    logger = WandbLogger(
        project=args.project_name,
    )
    logger.watch(model, log="all", log_freq=args.log_every_n_steps)
    save_dir = logger.experiment.dir
    os.makedirs(os.path.join(save_dir, "images"))

    ############## CALLBACKS ###############
    callbacks = [
        TQDMProgressBar(refresh_rate=5),
        ModelCheckpoint(
            monitor="loss",
            mode="min",
            dirpath=os.path.join(save_dir, "ckpt"),
            filename=(
                "[{step:06d}]-"
                "[{loss:.4f}]-"
                "[{content_loss:.4f}-"
                "[{style_loss:.4f}]]"
            ),
            auto_insert_metric_name=False,
            save_top_k=3,
            save_last=True,
            verbose=True,
            every_n_train_steps=1000,
        ),
    ]

    ############### TRAINER ################
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
    )

    ############# TRAIN START ##############
    dataloader = {
        "content": content_dataloader,
        "style": style_dataloader,
    }
    trainer.fit(model, dataloader)

    ############## ARTIFACTS ###############
    decoder_state_dict_path = os.path.join(logger.experiment.dir, "decoder.pth")
    state_dict = model.decoder.state_dict()
    torch.save(state_dict, decoder_state_dict_path)

    state_dict_path = os.path.join(logger.experiment.dir, "AdaIN.pth")
    state_dict = model.state_dict()
    torch.save(state_dict, state_dict_path)

    example_inputs = (
        torch.rand([1, 3, 256, 256]),
        torch.rand([1, 3, 256, 256]),
        torch.zeros(1),
    )

    torchscript_path = os.path.join(logger.experiment.dir, "AdaIN.pt.zip")
    model.to_torchscript(
        torchscript_path, "trace", example_inputs=example_inputs
    )

    onnx_path = os.path.join(logger.experiment.dir, "AdaIN.onnx")
    input_names = ["content", "style", "alpha"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size"},
        input_names[1]: {0: "batch_size"},
        input_names[2]: {0: "batch_size"},
        output_names[0]: {0: "batch_size"},
    }

    model.to_onnx(
        file_path=onnx_path,
        input_sample=example_inputs,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.onnx_opset_version,
        dynamic_axes=dynamic_axes,
    )

    if args.upload_artifacts:
        artifacts = wandb.Artifact(
            "Adaptive-Instance-Normalization", type="model"
        )
        artifacts.add_file(state_dict_path, "weight")
        artifacts.add_file(decoder_state_dict_path, "decoder_weight")
        artifacts.add_file(torchscript_path, "torchscript")
        artifacts.add_file(onnx_path, "onnx")
        logger.log_artifact(artifacts)


if __name__ == "__main__":
    main()
