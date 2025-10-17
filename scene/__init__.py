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

import json
import os
import random

import torch
from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.linear_model import LinearModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:

    linears: LinearModel

    def __init__(
        self,
        args: ModelParams,
        linears: LinearModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        fixed_init=False,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.linears = linears

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                fixed_init=fixed_init,
            )
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](
                args.source_path, args.white_background, args.eval, args.load_allres
            )
        elif os.path.exists(os.path.join(args.source_path, "colmap")):
            print("Found colmap folder, assuming Scannet++ data set!")
            scene_info = sceneLoadTypeCallbacks["Scannet"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if self.cameras_extent == 0:
            self.cameras_extent = 2.6
            print("Using default camera extent.")

        print("Camera extent: ", self.cameras_extent)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            point_cloud_path = os.path.join(
                self.model_path,
                "point_cloud/iteration_{}".format(self.loaded_iter),
                "point_cloud.ply",
            )
            if os.path.exists(point_cloud_path):
                self.linears.load_ply(point_cloud_path)
            else:
                # load checkpoint
                checkpoint_path = os.path.join(
                    self.model_path, "chkpnt{}.pth".format(self.loaded_iter)
                )
                (model_params, _) = torch.load(checkpoint_path)
                self.linears.restore(model_params, None)

                # save to ply
                self.linears.save_ply(point_cloud_path)
        else:
            self.linears.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.linears.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
