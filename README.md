# IDa-Det: An Information Discrepancy-aware Distillation for 1-bit Detectors
Implementation of our ECCV2022 paper.

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


Other models will be open-sourced successively.
