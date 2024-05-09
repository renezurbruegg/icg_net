<a name="readme-top"></a>
<div align="center">

  <h3 align="center">ICG-Net</h3>

  <p align="center">
    ICG-Net: A Unified Approach for Instance Centric Grasping
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
[![MIT License][license-shield]][license-url]

This repo contains the implementation of ICG-Net. For the **Benchmark and Checkpoints**, please refer to the  [Benchmark Repository](https://github.com/renezurbruegg/icg_benchmark).


[![Product Name Screen Shot][product-screenshot]](#)

## Getting Started

To get a local copy up and running follow these steps:

### Prerequisites
#### Installation

<details>
<summary>Pytorch 2.2, Cuda 12.1</summary>
To install MinkowskiEngine with Pytorch >= 2.0 and Cuda >= 12.0, you will need to install our patched version of MinkowskiEngine.
 
```bash
sudo apt-get install libopenblas-dev

export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
# Install conda environment
conda env create -f conda_cu121.yml
conda activate icg_cuda121

# Install Pytorch and Dependencies
pip install torch torchvision torchaudio torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html


# Install Patched MinkowskiEngine
git clone git@github.com:renezurbruegg/MinkowskiEngine.git
cd MinkowskiEngine

# Link conda cuda version for minkowski compilation
export CUDA_HOME=${CONDA_PREFIX}/envs/icg_cuda121
# Link cuda libraries.
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
# Link cuda headers. This might not be necessary for all systems.
sudo ln -s ${CONDA_PREFIX}/lib/libcudart.so.12 /usr/lib/libcudart.so
python setup.py install --force_cuda --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include --cuda_home=$CUDA_HOME
cd -

# Install third party requirements
cd icg_net/third_party/pointnet2
python setup.py install
cd -

# Install icg_net as pip package
pip install -e .
```
</details>

<details>
<summary>Pytorch 1.12 Cuda 11.3 </summary>
 
```bash
sudo apt-get install libopenblas-dev

# Install conda environment
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
conda env create -f conda_cu113.yml
conda activate icg_cuda113

# Install Pytorch and Dependencies
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html



# Install MinkowskiEngine
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

# Install third party requirements
cd icg_net/third_party/pointnet2
python setup.py install
cd -

# Install icg_net as pip package
pip install -e .
```
</details>

## Usage

```python

from icg_net import ICGNetModule, get_model
from icg_net.typing import ModelPredOut


# Load model
model: ICGNetModule = get_model(
    "icg_benchmark/data/51--0.656/config.yaml",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
pc, normals = # ... load pc

out: ModelPredOut = model(
    torch.from_numpy(np.asarray(o3dc.points)).float(),
    normals=torch.from_numpy(np.asarray(o3dc.normals)).float(),
    grasp_pts=grasp_pts,
    grasp_normals=grasp_normals,
    n_grasps=512,
    each_object=True,
    return_meshes=True,
    return_scene_grasps=True,
)

print(out)

```
 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the BSD-2 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Citing
If you use this code in your research, please cite the following paper:
```
@article{zurbrugg2024icgnet,
  title={ICGNet: A Unified Approach for Instance-Centric Grasping},
  author={Zurbr{\"u}gg, Ren{\'e} and Liu, Yifan and Engelmann, Francis and Kumar, Suryansh and Hutter, Marco and Patil, Vaishakh and Yu, Fisher},
  journal={arXiv preprint arXiv:2401.09939},
  year={2024}
}
```



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew -->

[license-url]: https://github.com/renezurbruegg/ICG-Net/blob/master/LICENSE.txt
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[product-screenshot]: docs/images/predictions.gif

<!-- [Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com  -->
