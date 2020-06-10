# NAS-FCOS: Fast Neural Architecture Search for Object Detection

This project hosts the train and inference code with pretrained model for implementing the NAS-FCOS algorithm for object detection, as presented in our paper:

    NAS-FCOS: Fast Neural Architecture Search for Object Detection;
    Ning Wang, Yang Gao, Hao Chen, Peng Wang, Zhi Tian, Chunhua Shen;
    In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2020.

The full paper is available at: [NAS-FCOS Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_NAS-FCOS_Fast_Neural_Architecture_Search_for_Object_Detection_CVPR_2020_paper.pdf). 

## Updates
* News: Accepted by CVPR 2020. (24/02/2020)
* Upload solver module to support self training. (06/02/2020)
* Support RetinaNet detector in NAS module (pretrained model coming soon). (06/02/2020)
* Update NAS head module, config files and pretrained model links. (07/01/2020)

## Required hardware
We use 4 Nvidia V100 GPUs. 


## Installation

This NAS-FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## Train
The train command line on coco train:

    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=1213 \
        tools/train_net.py --config-file "configs/search/R_50_NAS_retinanet.yaml"

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
Mobile_NAS_head | No | 34.4 | 34.7 | [download](https://mega.nz/#!ruIzSSxR!NIQEk-PG-aPYdSryRpSscE3pD81YJysVs6z1-o48-V0) | -
R_50_NAS | No | 38.5 | 38.9 | [download](https://pan.baidu.com/s/1-eH5Rs0KKGpx7nQa22vJOA) |f88u
R_50_NAS_head | No | 39.5 | 39.8 | [download](https://mega.nz/#!H7g0TQ6R!VL9jEjviVMSuPoOYeZceS8usnoy3bulZDrHc0QDIO6A) | -
R_101_NAS | Yes | 42.1 | 42.5 | [download](https://pan.baidu.com/s/1pRgVIsWtXdDea1EE23JGRg) | euuz
R_101_NAS_head | Yes | 42.8 | 43.0 | [download](https://mega.nz/#!uiJj3ICR!NdU3VaBtsdySFS0QezVpiV8Yz4h4CaqG63ST357860E) | -
R_101_X_32x8d_NAS | Yes | 43.4 | 43.7 | [download](https://pan.baidu.com/s/1tn6mfXKsaVH9-HBxQCNrTg) | 4cci

**Attention:** If the above model link cannot be downloaded normally, please refer to the link below.
[Mobile_NAS](https://mega.nz/#!Gu4DAS7K!Cp46jUVhOIvVhPUOtukrHKJfao_Pk5vAwaU_xz8haR0),
[Mobile_NAS_head](https://mega.nz/#!ruIzSSxR!NIQEk-PG-aPYdSryRpSscE3pD81YJysVs6z1-o48-V0),
[R_50_NAS](https://mega.nz/#!y34TGYbJ!Vv_k-GcGTW7A_F_Ov5f44PfzCfpK6oYrtS1ZIC9gFK8),
[R_50_NAS_head](https://mega.nz/#!H7g0TQ6R!VL9jEjviVMSuPoOYeZceS8usnoy3bulZDrHc0QDIO6A),
[R_101_NAS](https://mega.nz/#!Xqx1TS7S!MPiiasknw6M2aJjdR6SkevFFadgmJW8_TOJig_naXnE),
[R_101_NAS_head](https://mega.nz/#!uiJj3ICR!NdU3VaBtsdySFS0QezVpiV8Yz4h4CaqG63ST357860E)
[R_101_X_32x8d_NAS](https://mega.nz/#!qqpRUCaI!tj24t4tLWF_Qn56ZvdTkdxWzoXcP1gFEwgk4OK__Shw)


*All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* 


## Contributing to the project

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@InProceedings{Wang_2020_CVPR,
    author = {Wang, Ning and Gao, Yang and Chen, Hao and Wang, Peng and Tian, Zhi and Shen, Chunhua and Zhang, Yanning},
    title = {NAS-FCOS: Fast Neural Architecture Search for Object Detection},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 