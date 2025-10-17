#
# The original code is licensed under:
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
#
# The modifications of the code are licensed under the MIT License.
# It can also be found under the LICENSE.md file.
#

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
nerf_synthetic_scenes = [
    "mic",
    "chair",
    "ship",
    "materials",
    "lego",
    "drums",
    "ficus",
    "hotdog",
]
scannetpp_scenes = [
    "39f36da05b",
    "5a269ba6fe",
    "dc263dfbf0",
    "08bbbdcc3d",
    "fb564c935d",
]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--render_with_train", action="store_true")
parser.add_argument(
    "--iterations",
    type=int,
    nargs="+",
    default=[30000],
    help="Training iterations to run/save for each scene",
)
args, _ = parser.parse_known_args()

extra_training_args = ""

parser.add_argument("--mipnerf360", "-m360", type=str, default="")
parser.add_argument(
    "--mipnerf360_outdoor_scenes",
    "-m360o",
    type=str,
    nargs="*",
    default=mipnerf360_outdoor_scenes,
)
parser.add_argument(
    "--mipnerf360_indoor_scenes",
    "-m360i",
    type=str,
    nargs="*",
    default=mipnerf360_indoor_scenes,
)

parser.add_argument("--nerfsynthetic", "-ns", type=str, default="")
parser.add_argument(
    "--nerfsynthetic_scenes",
    "-nss",
    type=str,
    nargs="*",
    default=nerf_synthetic_scenes,
)

parser.add_argument(
    "--scannetpp", "-sp", type=str, default=""
)  # scannet train scenes
parser.add_argument(
    "--scannetpp_scenes", "-sps", type=str, nargs="*", default=scannetpp_scenes
)

args, unknown = parser.parse_known_args()

extra_training_args = " ".join(unknown)
print("Using additional training arguments: " + extra_training_args)

if args.mipnerf360 == "":
    print("Skipping MipNeRF360 Scenes since no data path is given.")
    mipnerf360_outdoor_scenes = []
    mipnerf360_indoor_scenes = []
else:
    mipnerf360_outdoor_scenes = args.mipnerf360_outdoor_scenes
    mipnerf360_indoor_scenes = args.mipnerf360_indoor_scenes

if args.nerfsynthetic == "":
    print("Skipping NeRF Synthetic Scenes since no data path is given.")
    nerf_synthetic_scenes = []
else:
    nerf_synthetic_scenes = args.nerfsynthetic_scenes

if args.scannetpp == "":
    print("Skipping ScanNet++ Scenes since no data path is given.")
    scannetpp_scenes = []
else:
    scannetpp_scenes = args.scannetpp_scenes

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(nerf_synthetic_scenes)
all_scenes.extend(scannetpp_scenes)

if len(all_scenes) == 0:
    print("Aborting, no scenes are given.")
    exit()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 " + extra_training_args + " "

    train_iterations = max(args.iterations)
    save_iterations = "--save_iterations " + " ".join(
        [str(iteration) for iteration in args.iterations]
    )

    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system(
            "python train.py --iterations "
            + str(train_iterations)
            + " -s "
            + source
            + " -i images_4 -m "
            + args.output_path
            + "/"
            + scene
            + common_args
            + " "
            + save_iterations
        )
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system(
            "python train.py --iterations "
            + str(train_iterations)
            + " -s "
            + source
            + " -i images_2 -m "
            + args.output_path
            + "/"
            + scene
            + common_args
            + " "
            + save_iterations
        )
    for scene in nerf_synthetic_scenes:
        source = args.nerfsynthetic + "/" + scene
        os.system(
            "python train.py --white_background --iterations "
            + str(train_iterations)
            + " -s "
            + source
            + " -m "
            + args.output_path
            + "/"
            + scene
            + common_args
            + " "
            + save_iterations
        )
    for scene in scannetpp_scenes:
        source = args.scannetpp + "/" + scene + "/dslr"
        os.system(
            "python train.py --iterations "
            + str(train_iterations)
            + " -s "
            + source
            + " --resolution 1 -m "
            + args.output_path
            + "/"
            + scene
            + common_args
            + " "
            + save_iterations
        )

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in nerf_synthetic_scenes:
        all_sources.append(args.nerfsynthetic + "/" + scene)
    for scene in scannetpp_scenes:
        all_sources.append(args.scannetpp + "/" + scene + "/dslr")

    common_args = (
        " --quiet --eval --skip_train"
        if not args.render_with_train
        else " --quiet --eval "
    )
    for scene, source in zip(all_scenes, all_sources):
        for iteration in args.iterations:
            os.system(
                "python render.py --iteration "
                + str(iteration)
                + " -s "
                + source
                + " -m "
                + args.output_path
                + "/"
                + scene
                + common_args
            )

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += '"' + args.output_path + "/" + scene + '" '

    os.system(
        "python metrics.py -m " + scenes_string
    )
    exit()
