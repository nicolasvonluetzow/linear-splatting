# LinPrim: Linear Primitives for Differentiable Volumetric Rendering
Nicolas von Lützow, Matthias Nießner<br>
| [Webpage](https://nicolasvonluetzow.github.io/LinPrim/) | [arXiv](https://arxiv.org/abs/2501.16312) | [Video](https://youtu.be/NRRlmFZj5KQ) |<br>

This is the official repository for the NeurIPS'25 paper "LinPrim: Linear Primitives for Differentiable Volumetric Rendering". LinPrim reconstructs 3D scenes by optimizing transparent polyhedra from known views resulting in an explicit and discrete representation.

This release includes minor cleanups compared to the paper version, resulting in slight improvements in rendering speed and memory efficiency without affecting reconstruction quality.

![Teaser image](assets/teaser_bike.jpeg)

 If you find LinPrim useful for your own work please consider citing:



<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{von2025linprim,
  title={LinPrim: Linear Primitives for Differentiable Volumetric Rendering},
  author={von L{\"u}tzow, Nicolas and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2501.16312},
  year={2025}
}</code></pre>
  </div>
</section>


## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
git clone git@github.com:nicolasvonluetzow/linear-splatting.git --recursive
```


## Setup
Our setup is tested only for Ubuntu Linux 20.04 - if you run into any issues please go check out the helpful information in the [3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting). We found a pip-based, step-by-step install (similarly used by Mip-Splatting) to work most consistently. Create the conda environment as follows:

```shell
conda create -y -n linear_splatting python=3.8
conda activate linear_splatting

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install plyfile ninja opencv-python

pip install submodules/diff-linear-rasterization
pip install submodules/simple-knn

```

## Running

Optimizing a scene with standard parameters works as follows:

```shell
python train.py -s <dataset path> -m <output path> 
```
or with MCMC-based densification which tends to yield better results but requires specifying a number of primitives:
```shell
python train.py -s <dataset path> -m <output path> --cap_max <primitive capacity> --use_mcmc --densify_until_iter 25_000 --densification_interval 100
```

We add some additional parameters e.g. to use MCMC-densification or adjust anti-aliasing filters. Other parameters remain mostly unchanged.
<details>
<summary><span style="font-weight: bold;">New Argument Highlights</span></summary>

  #### --use_mcmc
  Swap from 3DGS-style densification to the 3DGS-MCMC version. To replicate the results from the paper make sure to also adjust the densification_interval and densify_until_iter parameters.
  #### --cap_max
  Required when using MCMC. Specifies how many primitives will be used in total.
  #### --kernel_size
  Used kernel size of the 2D filter. Controls the minimal extent of primitives in screen space.
  #### --box_factor
  Used in the 3D filter. Controls the minimal size of the primitives.

</details>
<details>
<summary><span style="font-weight: bold;">Other Arguments</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --ray_jitter
  Adds per-ray subpixel jitter during training to reduce aliasing, as introduced in Mip-Splatting.
  #### --resample_gt_image
  Re-samples the ground-truth image with the same jitter offsets (use together with `--ray_jitter` to avoid blur).
  #### --sample_more_highres
  Upsamples the chance of picking high-resolution training views when multiple resolutions are available.
  #### --load_allres
  When working with the multi-scale Blender datasets, loads every resolution level instead of only the highest.
  #### --debug
  Enables debug mode if you experience errors. If the rasterizer fails, a ```dump``` file is created.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.025``` by default.
  #### --scaling_lr
  Distance learning rate, ```1e-4``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.00015``` by default.
  #### --densification_interval
  How frequently to densify, ```250``` (every 250 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --lambda_anisotropic
  Loss weight of an anisotropic loss. Unused by default.
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.
  #### --noise_lr
  Used in MCMC-densification. Controls the amount of noise added to primitive positions. ```5e5``` by default.
  #### --scale_reg
  Used in MCMC-densification. Regularizes primitive size. ```0.01/2.6``` by default.
  #### --opacity_reg
  Used in MCMC-densification. Regularizes primitive opacity. ```0.01``` by default.  

  TODO Mip-Splatting parameters
</details>

<br>


## Evaluation

We added some additional functionality to the [eval script](full_eval.py), which should simplify working with any of the datasets and scenes considered in the paper. Adding the path to a dataset will automatically run an evaluation for the respective scenes using the correct resolutions, train splits and background colors.

Any arguments given to the evaluation script that are not known will be passed to the training script.

Example usage for evaluating on ScanNet++ v2 and with MCMC@1M primitives:
```shell
python full_eval.py -sp <path to ScanNet> --use_mcmc --cap_max 1_000_000 --densify_until_iter 25_000 --densification_interval 100
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for full_eval.py</span></summary>

  #### --skip_training
  Flag to skip launching `train.py` for the selected scenes (triggers render/metrics only).
  #### --skip_rendering
  Flag to skip `render.py` execution (useful if renderings already exist on disk).
  #### --skip_metrics
  Flag to skip metric computation with `metrics.py`.
  #### --output_path
  Directory where scene-specific experiment folders are written, `./eval` by default.
  #### --render_with_train
  Flag that keeps the training split during rendering (otherwise rendering uses only the test split).
  #### --iterations
  Space-separated list of training iterations to target; the largest value is used for `--iterations` during training, and every value is forwarded to `--save_iterations` and subsequent rendering passes. Defaults to `30000`.
  #### --mipnerf360 / -m360
  Path to the root of a MipNeRF360 dataset. When omitted, the corresponding scenes are skipped.
  #### --mipnerf360_outdoor_scenes / -m360o
  Space-separated list of outdoor scene IDs to process when `--mipnerf360` is provided. Defaults to `bicycle flowers garden stump treehill`.
  #### --mipnerf360_indoor_scenes / -m360i
  Space-separated list of indoor scene IDs to process when `--mipnerf360` is provided. Defaults to `room counter kitchen bonsai`.
  #### --nerfsynthetic / -ns
  Path to the root of a NeRF Synthetic dataset. When omitted, those scenes are skipped.
  #### --nerfsynthetic_scenes / -nss
  Space-separated list of NeRF Synthetic scene IDs to evaluate. Defaults to `mic chair ship materials lego drums ficus hotdog`.
  #### --scannetpp / -sp
  Path to the root of a ScanNet++ dataset. When omitted, ScanNet++ scenes are skipped.
  #### --scannetpp_scenes / -sps
  Space-separated list of ScanNet++ scene IDs to evaluate. Defaults to `39f36da05b 5a269ba6fe dc263dfbf0 08bbbdcc3d fb564c935d`.

</details>

You can of course also run rendering and metrics manually using the individual scripts as in 3DGS.
```shell
python train.py -s <data path> --eval # Train with train/test split
python render.py -m <model path> # Generate renderings
python metrics.py -m <model path> # Compute error metrics on renderings
```


<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

</details>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

## Acknowledgements
We build upon the awesome open-source efforts of the following works - huge thanks to their authors and contributors!

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Mip-Splatting](https://github.com/autonomousvision/mip-splatting)
- [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc)
