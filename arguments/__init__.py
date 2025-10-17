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
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self._kernel_size = 0.1
        self._box_factor = 0.2
        self.ray_jitter = False
        self.resample_gt_image = False
        self.sample_more_highres = False
        self.load_allres = False  # for mipsplatting multi-scale
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


GLOBAL_LR_SCALE = 1e0


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = GLOBAL_LR_SCALE * 1.6e-4
        self.position_lr_final = GLOBAL_LR_SCALE * 1.6e-6
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = GLOBAL_LR_SCALE * 0.0025
        self.opacity_lr = GLOBAL_LR_SCALE * 0.025
        self.rotation_lr = GLOBAL_LR_SCALE * 0.001
        self.scaling_lr = GLOBAL_LR_SCALE * 1e-4
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_anisotropic = 0.0 
        self.densification_interval = 250
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.00015
        self.random_background = False
        self.grad_acc_steps = 1
        self.prune_min_opacity = 0.025

        # MCMC parameters, not used if MCMC is not enabled
        self.use_mcmc = False
        self.noise_lr = 5e5
        self.scale_reg = (
            0.01 / 2.6
        )  # account for size difference between dists and gs scales
        self.opacity_reg = 0.01
        self.cap_max = 1e6  # 1 million
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    cfgfile_string = cfgfile_string.split("\n")[0]
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
