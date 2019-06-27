# NAS-FCOS: Fast Neural Architecture Search for Object Detection

This project hosts the inference code and model for implementing the NAS-FCOS algorithm for object detection, as presented in our paper:

    NAS-FCOS: Fast Neural Architecture Search for Object Detection;
    Ning Wang, Yang Gao, Hao Chen, Peng Wang, Zhi Tian, Chunhua Shen;
    arXiv preprint arXiv:1906.04423 (2019).

The full paper is available at: [https://arxiv.org/abs/1906.04423](https://arxiv.org/abs/1906.04423). 


## Required hardware
We use 4 Nvidia V100 GPUs. 


## Installation

This NAS-FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.


## Inference
The inference command line on coco minival split:

    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        tools/test_net.py --config-file "configs/search/R_50_NAS_densebox.yaml"

Please note that:
1) If your model's name is different, please replace `models/R-50-NAS.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/search](configs/search)) and `MODEL.WEIGHT` to its weights file.

For your convenience, we provide the following trained models (more models are coming soon).

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link | Fetch Code
--- |:---:|:---:|:--:|:---:|:---:|
Mobile_NAS | No | 32.6 | 33.1 | [download](https://pan.baidu.com/s/1wx1qeiIVo64d51zyiJAauQ) | 3dm9 
R_50_NAS | No | 38.5 | 38.9 | [download](https://pan.baidu.com/s/1-eH5Rs0KKGpx7nQa22vJOA) |f88u
R_101_NAS | Yes | 42.1 | 42.5 | [download](https://pan.baidu.com/s/1pRgVIsWtXdDea1EE23JGRg) | euuz
R_101_X_32x8d_NAS | Yes | 43.4 | 43.7 | [download](https://pan.baidu.com/s/1tn6mfXKsaVH9-HBxQCNrTg) | 4cci

*All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* 


## Contributing to the project

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{wang2019nasfcos,
  title   =  {{NAS-FCOS}: Fast Neural Architecture Search for Object Detection},
  author  =  {Wang, Ning and Gao, Yang and Chen, Hao and Wang, Peng and Tian, Zhi and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:1906.04423},
  year    =  {2019}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 