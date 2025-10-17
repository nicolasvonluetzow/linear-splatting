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
# Parts of this code were taken from 3DGS-MCMC and Mip-Splatting:
# 3DGS-MCMC: https://github.com/ubc-vision/3dgs-mcmc
# Mip-Splatting: https://github.com/autonomousvision/mip-splatting
#

import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torchvision
from arguments import ModelParams, OptimizationParams, PipelineParams
from linear_renderer import render
from scene import LinearModel, Scene
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing="xy")
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()

    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= width - 1
    id_coords[..., 1] /= height - 1
    id_coords = id_coords * 2 - 1

    image = torch.nn.functional.grid_sample(
        image[None], id_coords[None], align_corners=True, padding_mode="border"
    )[0]
    return image


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Save all the parameters
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))
        cfg_log_f.write("\n")
        cfg_log_f.write(str(Namespace(**vars(opt))))
        cfg_log_f.write("\n")
        cfg_log_f.write(str(Namespace(**vars(pipe))))

    if "box_factor" in vars(dataset):
        linears = LinearModel(dataset.sh_degree, dataset.box_factor)
    else:
        linears = LinearModel(dataset.sh_degree)
    scene = Scene(dataset, linears)
    linears.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        linears.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    allCameras = train_cameras + test_cameras

    linears.compute_3D_filter(cameras=train_cameras)

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(train_cameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    # # Sanity check, save image of initialization
    # viewpoint_cam = scene.getTrainCameras()[0]
    # bg = torch.rand((3), device="cuda") if opt.random_background else background
    # rendered = render(
    #     viewpoint_cam, linears, pipe, bg, kernel_size=dataset.kernel_size
    # )["render"]
    # torchvision.utils.save_image(
    #     rendered, os.path.join(dataset.model_path, f"rendering_init.png")
    # )
    # # Save GT image
    # gt_image = viewpoint_cam.original_image.cuda()
    # torchvision.utils.save_image(
    #     gt_image, os.path.join(dataset.model_path, f"gt_init.png")
    # )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="Training progress", ncols=0
    )
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         # if custom_cam != None:
        #             # net_image = render(custom_cam, linears, pipe, background, scaling_modifer)["render"]
        #             # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = linears.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            linears.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = train_cameras[
                highresolution_index[randint(0, len(highresolution_index) - 1)]
            ]

        # print(f"Optimizing for viewpoint {viewpoint_cam.image_name}")

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dataset.ray_jitter:
            subpixel_offset = (
                torch.rand(
                    (
                        int(viewpoint_cam.image_height),
                        int(viewpoint_cam.image_width),
                        2,
                    ),
                    dtype=torch.float32,
                    device="cuda",
                )
                - 0.5
            )
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam,
            linears,
            pipe,
            bg,
            kernel_size=dataset.kernel_size,
            subpixel_offset=subpixel_offset,
        )
        image, viewspace_point_tensor, visibility_filter = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
        )
        radii = render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss += opt.lambda_anisotropic * linears.anisotropic_loss(visibility_filter)

        # mcmc losses
        if opt.use_mcmc:
            loss += opt.scale_reg * torch.abs(linears.get_dist).mean()
            loss += opt.opacity_reg * torch.abs(linears.get_opacity).mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_fac = 0.4
            ema_loss_for_log = ema_fac * loss.item() + (1 - ema_fac) * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                linears,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, dataset.kernel_size),
            )  # , rand_linears)
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Linears".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                if not opt.use_mcmc:
                    # Keep track of max radii in image-space for pruning
                    linears.max_radii2D[visibility_filter] = torch.max(
                        linears.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    linears.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    if not opt.use_mcmc:
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )
                        linears.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.prune_min_opacity,
                            scene.cameras_extent,
                            size_threshold,
                            tb_writer,
                            iteration,
                        )
                    else:
                        dead_mask = (linears.get_opacity <= 0.005).squeeze(-1)
                        linears.relocate_gs(dead_mask=dead_mask)
                        linears.add_new_gs(cap_max=opt.cap_max)
                    linears.compute_3D_filter(cameras=train_cameras)

                if not opt.use_mcmc:
                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        linears.reset_opacity()

            if iteration & 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    linears.compute_3D_filter(cameras=train_cameras)

            # Optimizer step
            if iteration < opt.iterations and iteration % opt.grad_acc_steps == 0:
                linears.optimizer.step()
                linears.optimizer.zero_grad(set_to_none=True)

                if opt.use_mcmc:
                    # add some noise to the positions
                    linears.add_noise_to_xyz(opt.noise_lr, xyz_lr)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (linears.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    # print_primitive_info(linears)


def print_primitive_info(primitives):
    info_string = (
        f"#prims {primitives.get_xyz.shape[0]}"
        + f" - opa {primitives.get_opacity.mean().detach().cpu().numpy():.3f} ({primitives.get_opacity.min().detach().cpu().numpy():.2f}, {primitives.get_opacity.max().detach().cpu().numpy():.2f})"
        + f" - size {primitives.get_size.mean().detach().cpu().numpy():.3f} ({primitives.get_size.min().detach().cpu().numpy():.2f}, {primitives.get_size.max().detach().cpu().numpy():.2f})"
        + f" - 3D filter {primitives.get_dist_with_3D_filter.mean().detach().cpu().numpy():.3f} ({primitives.get_dist_with_3D_filter.min().detach().cpu().numpy():.2f}, {primitives.get_dist_with_3D_filter.max().detach().cpu().numpy():.2f})"
    )

    print(info_string)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def pixel_wise_error(pred, gt):
    pix_l1 = torch.abs(pred - gt)
    return pix_l1


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    primitives,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

        tb_writer.add_scalar(
            "stats/size_mean", primitives.get_size.mean().item(), iteration
        )
        tb_writer.add_scalar(
            "stats/size_max", primitives.get_size.max().item(), iteration
        )
        tb_writer.add_scalar(
            "stats/opacity_mean", primitives.get_opacity.mean().item(), iteration
        )

        tb_writer.add_scalar(
            "stats/num_primitives", primitives.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )
        # validation_configs = ({'name': 'train', 'cameras' : scene.getTrainCameras()},)

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.linears, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 3):
                        torchvision.utils.save_image(
                            image,
                            os.path.join(
                                scene.model_path,
                                f"{config['name']}_{viewpoint.image_name}_it{iteration}_rend.png",
                            ),
                        )
                        if iteration == testing_iterations[0]:
                            torchvision.utils.save_image(
                                gt_image,
                                os.path.join(
                                    scene.model_path,
                                    f"{config['name']}_{viewpoint.image_name}_gt.png",
                                ),
                            )

                    # Error image
                    if tb_writer and (idx == 0):
                        pix_loss = pixel_wise_error(
                            image.clone().detach(), gt_image.clone().detach()
                        )
                        pix_loss = (
                            torch.abs(pix_loss).mean(dim=0).unsqueeze(0)
                            / pix_loss.max()
                        )
                        # tb_writer.add_image(config['name'] + "_view_{}/error".format(viewpoint.image_name), pix_loss, global_step=iteration)
                        torchvision.utils.save_image(
                            pix_loss,
                            os.path.join(
                                scene.model_path,
                                f"{config['name']}_{viewpoint.image_name}_err.png",
                            ),
                        )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
        print_primitive_info(primitives)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.linears.get_opacity, iteration)
        #     tb_writer.add_histogram("scene/size_histogram", scene.linears.get_size, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    # For dev
    np.set_printoptions(precision=2)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
