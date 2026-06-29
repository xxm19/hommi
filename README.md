# [RSS 2026] HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations
[[Project page]](https://hommi-robot.github.io)
[[Paper]](https://arxiv.org/abs/2603.03243)
[[Video]](https://youtu.be/0rPr9DbPfuo?si=mssJLSQCmZ-V3hzW)
[[Hardware]](https://drive.google.com/drive/folders/11eYyGpZgONSzI9i9WdD2V2F-GlcFdUVo?usp=share_link)

[Xiaomeng Xu](https://xxm19.github.io/)<sup>1,2</sup>,
[Jisang Park](https://jisangpark.com)<sup>1</sup>,
[Han Zhang](https://doublehan07.github.io/)<sup>1</sup>,
[Eric Cousineau](https://www.eacousineau.com)<sup>2</sup>,
[Aditya Bhat](https://www.linkedin.com/in/mradityabhat/)<sup>2</sup>,
[Jose Barreiros](https://www.josebarreiros.com)<sup>2</sup>,
[Dian Wang](https://www.dianwang.io/)<sup>1</sup>,
[Jeannette Bohg](https://web.stanford.edu/~bohg/)<sup>1</sup>,
[Shuran Song](https://shurans.github.io/)<sup>1</sup>

<sup>1</sup>Stanford University,  
<sup>2</sup>Toyota Research Institute

<img width="100%" src="teaser.jpg">

## 🛠️ Installation

Clone this repo and clone submodules
```bash
git clone --recurse-submodules git@github.com:xxm19/hommi.git
```

### Setup mamba env:
```bash
# this automatically does editable install of `hommi` package from local source
mamba env create -f environment.yml
mamba activate hommi
conda install -c dglteam/label/th24_cu124 dgl
```

### Set PYTHONPATH for local dependencies
This repo depends on local sources from `deps/universal_manipulation_interface` (e.g. `diffusion_policy`, `umi`) and the RBY1 WBC submodule at `deps/rby1-wbc`.
Add this to your shell config (e.g. `~/.zshrc`), then reload the shell:
```bash
export HOMMI_ROOT=/path/to/hommi
export HOMMI_UMI_ROOT="$HOMMI_ROOT/deps/universal_manipulation_interface"
export HOMMI_RBY1_WBC_ROOT="$HOMMI_ROOT/deps/rby1-wbc"
export PYTHONPATH="$HOMMI_UMI_ROOT:$HOMMI_RBY1_WBC_ROOT:$HOMMI_ROOT:${PYTHONPATH}"
```

## Data Collection
Download iPhone mount CAD file [here](https://drive.google.com/drive/folders/11eYyGpZgONSzI9i9WdD2V2F-GlcFdUVo?usp=share_link)

Install and setup iPhone app for data collection [here](https://github.com/real-stanford/iPhUMI)

## Process HoMMI demonstrations
```bash
cd hommi/demonstration_processing
```

### Group, time align and visualize demonstrations
```bash
python process_demos_iphone.py group.iphone_dir=iphone_data filters.session_name=test
```

### Create a data session
Set input session filters and output session name in yaml config file
```bash
python create_session_iphone.py input_session_filters=test output_session_name=test
```

### Build a dataset from the session
```bash
python build_umi_dataset_iphone.py session_dir=tmp_sessions/test
# generates dataset_plan.pkl and dataset.zarr.zip
```

## Train a HoMMI policy
```bash
python hommi/train_network/train.py --config-name=umi_policy_dit model=diffusion_dit_3d task=ego3d_lookat_policy task.dataset_path=dataset.zarr.zip
```

multi-gpu training:
```bash
accelerate launch --mixed_precision bf16 --num_processes 8 --multi_gpu --gradient_accumulation_steps 4 -- /path/to/train.py [args...]
```

## Deployment
### Robot, camera, and WBC setup
Deployment code lives under `hommi/deployment`:

- Camera setup: see `hommi/deployment/camera/README.md`. edit `hommi/deployment/config/camera.yaml` for camera serials, mock mode, frame rate, and binning. use `python -m hommi.deployment.camera_stream_viewer` to inspect streams.
- WBC/WBIK setup: initialize `deps/rby1-wbc`, then follow `deps/rby1-wbc/control/README.md` to build the RBY1 realtime controller pybind module and configure `deps/rby1-wbc/config/wbc.yaml` / `deps/rby1-wbc/config/wbik.yaml`.

### Policy rollout
Run these in separate terminals (in the same hommi conda environment):

```bash
python -m hommi.deployment.policy_server -i checkpoint_path
python -m hommi.deployment.rby1_wbc_policy
```

## 📜 Citation
```console
@article{xu2026hommi,
  title={Hommi: Learning whole-body mobile manipulation from human demonstrations},
  author={Xu, Xiaomeng and Park, Jisang and Zhang, Han and Cousineau, Eric and Bhat, Aditya and Barreiros, Jose and Wang, Dian and Bohg, Jeannette and Song, Shuran},
  journal={arXiv preprint arXiv:2603.03243},
  year={2026}
}
```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](https://github.com/xxm19/HoMMI/blob/main/LICENSE) for more details.

## 🙏 Acknowledgement
- Our diffusion policy implementation is adapted from [UMI](https://github.com/real-stanford/universal_manipulation_interface).
- Our 3D visual encoder implementation is adapted from [Adapt3R](https://github.com/pairlab/Adapt3R).
