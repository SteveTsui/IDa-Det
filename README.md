# IDa-Det: An Information Discrepancy-aware Distillation for 1-bit Detectors
Pytorch implementation of our paper ["IDa-Det: An Information Discrepancy-aware Distillation for 1-bit Detectors"](http://arxiv.org/abs/2210.03477) accepted by ECCV2022.

### Tips
Any problem, please contact the first author (Email: shengxu@buaa.edu.cn). 

Our code is heavily borrowed from DeFeat (https://github.com/ggjy/DeFeat.pytorch/) and based on MMDetection (https://github.com/open-mmlab/mmdetection).


### Environments
- Python 3.7
- MMDetection 2.x
- This repo uses: `mmdet-v2.0` `mmcv-0.5.6` `cuda 10.1`

### VOC Results

Pretrained model is here: [GoogleDrive](https://drive.google.com/drive/folders/1I0RlAiLe-KJuorXq4lLzeSgJUb3inT2J?usp=sharing)

**Notes:**

- Faster RCNN based model
- Batch: sample_per_gpu x gpu_num

| Model  | Batch | Lr schd | box AP | Model | Log |
|:-----:|:-----:|:-------:|:------:|:-----:|:---:|
| R101  |  4x2  | 0.01    | 81.9  |[GoogleDrive](https://drive.google.com/drive/folders/1I0RlAiLe-KJuorXq4lLzeSgJUb3inT2J?usp=sharing) |     |
| R101-BiR18| 4x1  | 0.004    | 76.9 | [GoogleDrive](https://drive.google.com/drive/folders/1I0RlAiLe-KJuorXq4lLzeSgJUb3inT2J?usp=sharing)|   |

If you find this work useful in your research, please consider to cite:

```
@inproceedings{xu2022ida,
  title={IDa-Det: An Information Discrepancy-Aware Distillation for 1-Bit Detectors},
  author={Xu, Sheng and Li, Yanjing and Zeng, Bohan and Ma, Teli and Zhang, Baochang and Cao, Xianbin and Gao, Peng and L{\"u}, Jinhu},
  booktitle={European Conference on Computer Vision},
  pages={346--361},
  year={2022},
  organization={Springer}
}
```
