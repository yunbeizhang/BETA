"""
Entry point for BETA test-time adaptation experiments.

Usage example (ViT-B/16 on ImageNet-C, best hyper-parameters):

    python main.py \
        --algorithm beta \
        --model vitb16 \
        --local_helper vits16 \
        --pad_size 16 \
        --alpha 0.4 \
        --margin_e0 0.9 \
        --bvr_lr 0.01 \
        --norm_lr 0.00002 \
        --kl_weight 50 \
        --batch_size 64 \
        --corruption all \
        --seed 2020

See `main.sh` for the reference configuration used in the paper.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask

from tta_library.beta import BETA, collect_params, configure_model
from models.prompter import PadPrompter, ProbFuser, PadVR
from models.prepare_model import prepare_pretrained_model

from utils.metrics import ECELoss

# Baselines
import tta_library.tent as tent
import tta_library.sar as sar
import tta_library.cotta as cotta
import tta_library.eata as eata
from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.foa import FOA
from tta_library.lame import LAME


ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

BETA_ALGOS = {"beta"}  # algorithms whose forward returns (output, model_out, local_out)


def validate_adapt(val_loader, model, args, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    model_top1 = AverageMeter("Model Acc@1", ":6.2f")
    local_top1 = AverageMeter("Local Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, model_top1, local_top1],
        prefix="Test: ",
    )

    outputs_list, targets_list = [], []
    end = time.time()
    for i, dl in enumerate(val_loader):
        images, target = dl[0], dl[1]
        if args.gpu is not None:
            images = images.cuda()
        if torch.cuda.is_available():
            target = target.cuda()

        if args.algorithm in BETA_ALGOS:
            output, model_outputs, local_outputs = model(images)
        else:
            output = model(images)

        outputs_list.append(output.detach().cpu())
        targets_list.append(target.detach().cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if args.algorithm in BETA_ALGOS:
            m_acc1, _ = accuracy(model_outputs, target, topk=(1, 5))
            l_acc1, _ = accuracy(local_outputs, target, topk=(1, 5))
            model_top1.update(m_acc1[0], images.size(0))
            local_top1.update(l_acc1[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(val_loader) - 1:
            logger.info(progress.display(i))

    outputs_list = torch.cat(outputs_list, dim=0).numpy()
    targets_list = torch.cat(targets_list, dim=0).numpy()

    # BETA & LAME output probabilities; others return logits.
    logits = args.algorithm not in (BETA_ALGOS | {"lame"})
    ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits)
    return top1.avg, top5.avg, ece_avg


def build_beta(net, local_helper, args):
    prompter = PadPrompter(pad_size=args.pad_size)
    probfuser = ProbFuser(alpha=args.alpha, learnable=False)
    local_helper, prompter, probfuser = configure_model(local_helper, prompter, probfuser)
    vr_params, _, norm_params, _, _, _ = collect_params(local_helper, prompter, probfuser)

    optimizer_norm = torch.optim.SGD(norm_params, args.norm_lr, momentum=0.9)
    optimizer_vr = torch.optim.AdamW(vr_params, args.bvr_lr)

    return BETA(
        model=net,
        local_helper=local_helper,
        prompter=prompter,
        probfuser=probfuser,
        optimizer_vr=optimizer_vr,
        optimizer_norm=optimizer_norm,
        steps=args.steps,
        e_margin=args.margin_e0 * math.log(1000),
        kl_weight=args.kl_weight,
    )


def build_adapt_model(net, local_helper, args):
    a = args.algorithm
    if a == "beta":
        assert local_helper is not None, "BETA requires --local_helper"
        return build_beta(net, local_helper, args)
    if a == "tent":
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        optimizer = torch.optim.SGD(params, args.norm_lr, momentum=0.9)
        return tent.Tent(net, optimizer)
    if a == "eata":
        net = eata.configure_model(net)
        params, _ = eata.collect_params(net)
        optimizer = torch.optim.SGD(params, 2.5e-4, momentum=0.9)
        return eata.EATA(net, optimizer, e_margin=0.4 * math.log(1000), d_margin=0.05)
    if a == "sar":
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        optimizer = SAM(params, torch.optim.SGD, lr=args.norm_lr, momentum=0.9)
        return sar.SAR(net, optimizer, margin_e0=args.margin_e0)
    if a == "cotta":
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9)
        return cotta.CoTTA(net, optimizer, steps=1, episodic=False)
    if a == "lame":
        return LAME(net)
    if a == "t3a":
        return T3A(net, 1000, 20).cuda()
    if a == "foa":
        from models.vpt import PromptViT
        net = PromptViT(net, args.num_prompts).cuda()
        adapt = FOA(net, args.fitness_lambda)
        args.corruption = "original"
        _, train_loader = prepare_test_data(args)
        adapt.obtain_origin_stat(train_loader)
        return adapt
    if a == "random_vr":
        return PadVR(net, pad_size=args.pad_size, input_size=224, output_size=224)
    if a == "no_adapt":
        return net
    raise NotImplementedError(f"Unknown algorithm: {a}")


def get_args():
    p = argparse.ArgumentParser(description="BETA: TTA for Black-Box Models")

    # Data
    p.add_argument("--data", default="/data/imagenet")
    p.add_argument("--data_corruption", default="/data/imagenet-c")
    p.add_argument("--data_rendition", default="/data/imagenet-r")
    p.add_argument("--data_v2", default="/data/imagenet")
    p.add_argument("--data_sketch", default="/data/imagenet")
    p.add_argument("--data_adv", default="/data/imagenet")

    # Loader / run
    p.add_argument("--seed", default=2020, type=int)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--if_shuffle", default=True, type=bool)
    p.add_argument("--debug", default=False, type=bool)

    # Algorithm
    p.add_argument("--algorithm", default="beta", type=str,
                   choices=["beta", "tent", "eata", "sar", "cotta", "lame", "t3a",
                            "foa", "random_vr", "no_adapt"])

    # Dataset
    p.add_argument("--level", default=5, type=int)
    p.add_argument("--corruption", default="all", type=str)

    # Models
    p.add_argument("--model", default="vitb16",
                   help="Target (black-box) architecture: vitb16 / r50 / ...")
    p.add_argument("--local_helper", default="vits16",
                   help="Local helper architecture for BETA (e.g., vits16 / r18).")

    # BETA hyper-parameters
    p.add_argument("--pad_size", default=16, type=int,
                   help="PadPrompter border size (paper default: 16).")
    p.add_argument("--bvr_lr", default=0.01, type=float,
                   help="Learning rate for the prompter.")
    p.add_argument("--norm_lr", default=2e-5, type=float,
                   help="Learning rate for normalization layers of the local helper.")
    p.add_argument("--alpha", default=0.4, type=float,
                   help="Fusion weight between local and black-box probabilities.")
    p.add_argument("--margin_e0", default=0.9, type=float,
                   help="Entropy margin multiplier e0 (final margin = e0*log(1000)).")
    p.add_argument("--kl_weight", default=50, type=float,
                   help="KL consistency weight λ.")
    p.add_argument("--steps", default=1, type=int)

    # Baseline-specific
    p.add_argument("--num_prompts", default=3, type=int)
    p.add_argument("--fitness_lambda", default=0.4, type=float)

    # Logging
    p.add_argument("--output", default="./outputs")
    p.add_argument("--tag", default="", type=str)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    return p.parse_args()


def main():
    args = get_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    args.output = os.path.join(args.output, args.algorithm + args.tag)
    os.makedirs(args.output, exist_ok=True)
    logger = get_logger(
        name="beta",
        output_directory=args.output,
        log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-log.txt",
        debug=False,
    )
    logger.info(args)

    corruptions = [args.corruption] if args.corruption != "all" else ALL_CORRUPTIONS

    # Build models
    net = prepare_pretrained_model(args.model).cuda().eval()
    net.requires_grad_(False)

    local_helper = None
    if args.algorithm == "beta" and args.local_helper:
        local_helper = prepare_pretrained_model(args.local_helper).cuda().eval()
        local_helper.requires_grad_(False)

    adapt_model = build_adapt_model(net, local_helper, args)

    if args.wandb:
        import wandb
        wandb.init(project="BETA", name=args.algorithm + args.tag)
        wandb.config.update(args)

    corrupt_acc, corrupt_ece = [], []
    for c in corruptions:
        args.corruption = c
        logger.info(f"Running on corruption: {c}")

        if c == "rendition":
            adapt_model.imagenet_mask = imagenet_r_mask
        else:
            adapt_model.imagenet_mask = None

        _, val_loader = prepare_test_data(args)

        torch.cuda.empty_cache()
        top1, top5, ece = validate_adapt(val_loader, adapt_model, args, logger)
        logger.info(f"{c} + {args.algorithm} | Top-1: {top1:.3f} | "
                    f"Top-5: {top5:.3f} | ECE: {ece * 100:.3f}")
        corrupt_acc.append(top1)
        corrupt_ece.append(ece)

        if args.algorithm in BETA_ALGOS:
            logger.info(f"Adapt info: {adapt_model.info}")

        if args.algorithm not in ("no_adapt", "random_vr"):
            adapt_model.reset()

    if corrupt_acc:
        logger.info(f"Mean acc:  {sum(corrupt_acc) / len(corrupt_acc):.3f}")
        logger.info(f"Mean ECE: {sum(corrupt_ece) / len(corrupt_ece) * 100:.3f}")


if __name__ == "__main__":
    main()
