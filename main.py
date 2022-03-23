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
    # logger.watch(model, log="all", log_freq=args.log_every_n_steps)
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
    ########### BUILD DATASET ############
    content_dataloader = build_dataloader(args, args.content_root_dir)
    style_dataloader = build_dataloader(args, args.style_root_dir)

    ############# TRAIN START ##############
    trainer.fit(
        model,
        {
            "content": content_dataloader,
            "style": style_dataloader,
        },
    )

    ############## ARTIFACTS MODEL ###############
    model = model.cpu().eval()
    model_weights_path = os.path.join(logger.experiment.dir, "model.pth")
    state_dict = model.state_dict()
    torch.save(state_dict, model_weights_path)

    example_inputs = (
        torch.rand([1, 3, 256, 256]),
        torch.rand([1, 3, 256, 256]),
        torch.zeros(1),
    )

    models_ts_path = os.path.join(logger.experiment.dir, "model.pt.zip")
    model.to_torchscript(
        models_ts_path,
        "trace",
        example_inputs=example_inputs,
    )

    model_onnx_path = os.path.join(logger.experiment.dir, "AdaIN.onnx")
    input_names = ["content", "style", "alpha"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        input_names[1]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
    }

    model.to_onnx(
        file_path=model_onnx_path,
        input_sample=example_inputs,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.onnx_opset_version,
        dynamic_axes=dynamic_axes,
    )

    ############## ARTIFACTS ENCODER ###############
    encoder_example_inputs = torch.rand([1, 3, 256, 256])
    encoder = nn.Sequential(*list(encoder[:31])).cpu().eval()
    encoder_ts_path = os.path.join(logger.experiment.dir, "encoder.pt.zip")
    encoder = torch.jit.trace(encoder, example_inputs=encoder_example_inputs)
    encoder.save(encoder_ts_path)

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
    }
    encoder_onnx_path = os.path.join(logger.experiment.dir, "encoder.onnx")
    torch.onnx.export(
        encoder,
        args=encoder_example_inputs,
        f=encoder_onnx_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.onnx_opset_version,
        dynamic_axes=dynamic_axes,
    )

    ############## ARTIFACTS DECODER ###############
    decoder = model.decoder.cpu().eval()

    decoder_weights_path = os.path.join(logger.experiment.dir, "decoder.pth")
    state_dict = decoder.state_dict()
    torch.save(state_dict, decoder_weights_path)

    decoder_example_inputs = encoder(encoder_example_inputs)
    decoder_ts_path = os.path.join(logger.experiment.dir, "decoder.pt.zip")
    decoder = torch.jit.trace(decoder, example_inputs=decoder_example_inputs)
    decoder.save(decoder_ts_path)

    decoder_onnx_path = os.path.join(logger.experiment.dir, "decoder.onnx")
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
    }
    torch.onnx.export(
        decoder,
        args=decoder_example_inputs,
        f=decoder_onnx_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.onnx_opset_version,
        dynamic_axes=dynamic_axes,
    )

    ############## ARTIFACTS ADAIN ###############
    adain_ts_path = os.path.join(logger.experiment.dir, "adain.pt.zip")
    adain = torch.jit.script(model.AdaIN.eval())
    adain.save(adain_ts_path)

    adain_example_inputs = (decoder_example_inputs, decoder_example_inputs)
    adain_onnx_path = os.path.join(logger.experiment.dir, "adain.onnx")
    input_names = ["content", "style"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        input_names[1]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
    }
    torch.onnx.export(
        adain,
        args=adain_example_inputs,
        f=adain_onnx_path,
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
        artifacts.add_file(model_weights_path, "model_weight")
        artifacts.add_file(models_ts_path, "model_torchscript")
        artifacts.add_file(model_onnx_path, "model_onnx")

        artifacts.add_file(encoder_ts_path, "encoder_torchscript")
        artifacts.add_file(encoder_onnx_path, "encoder_onnx")

        artifacts.add_file(decoder_weights_path, "decoder_weight")
        artifacts.add_file(decoder_ts_path, "decoder_torchscript")
        artifacts.add_file(decoder_onnx_path, "decoder_onnx")

        artifacts.add_file(adain_ts_path, "adain_torchscript")
        artifacts.add_file(adain_onnx_path, "adain_onnx")
        logger.log_artifact(artifacts)


if __name__ == "__main__":
    main()
