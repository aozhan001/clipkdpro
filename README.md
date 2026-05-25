# CLIP-KD
This repository contains the source code of CLIP-KD [CLIP-KD: An Empirical Study of CLIP Model Distillation].

## Install
```
pip install -r requirements-training.txt
pip install -r requirements-test.txt
```
## Dataset preparation

### Conceptual Captions 3M 

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. The names of the columns are passed as an argument to `main.py`.

The script `src/data/gather_cc.py` will collect the Conceptual Captions 3M images. First, download the [Conceptual Captions 3M URLs](https://ai.google.com/research/ConceptualCaptions/download) and then run the script from our repository:
For easy notation, we rename `Train_GCC-training` as `cc3m_train`, and `Validation_GCC-1.1.0-Validation` as `cc3m_val`.
```bash
python src/data/gather_cc.py [path/to/cc3m/images/] [path/to/cc3m_train.tsv] [path/to/cc3m_val.tsv]
```

Our downloaded CC3M training set contains 2.89M images, and our CC3M validation set contains 13K images.


The generated `cc3m_train.csv` is:
```
title   filepath
XXXXXX  train/X/X.jpg
...     ...
```

The generated `cc3m_val.csv` is:
```
title   filepath
XXXXXX  val/X/X.jpg
...     ...
```

### Conceptual 12M 
The script `src/data/gather_cc12m.py` will collect the Conceptual 12M images. First, download the [Conceptual 12M URLs](https://storage.googleapis.com/conceptual_12m/cc12m.tsv) and then run the script from our repository:

```bash
python src/data/gather_cc12m.py [path/to/cc12m/images/] [path/to/cc12m.tsv]
```
The generated `cc12m.csv` is:
```
title   filepath
XXXXXX  train/X/X.jpg
...     ...
```

Our downloaded CC12M training set contains 9.97M images.



## Distill CLIP models

### Distillation with different strategies
The teacher is pretrained on CC3M+12M. Students are distilled on CC3M+12M.
| Role | Network |Method | ImageNet Acc| Train script |
| :----: | :----: | :----:  |:----:  |:----: |
|  Teacher | ViT-B/16|-| 36.99 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_B_16_baseline.sh)|
|  Student | ViT-T/16|Baseline|30.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)|
|  Student | ViT-T/16| +CRD |31.94|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/CRD.sh)|
|  Student | ViT-T/16| +FD | 34.23|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/FD.sh)|
|  Student | ViT-T/16| +MFD | 34.09|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/MFD.sh)|
|  Student | ViT-T/16| +GD |31.54|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/GD.sh)|
|  Student | ViT-T/16| +ICL |33.11|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/ICL.sh)|
|  Student | ViT-T/16| +AFD |31.42|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/methods/AFD.sh)|



### Supervised by ViT-B/16 as the teacher
The teacher is pretrained on CC3M+12M. Students are distilled on CC3M+12M.
| Role | Network |Method | ImageNet Acc| Train script | Download |
| :----:  | :----:  | :----:  |:----: |:----: | :----: |
|  Teacher | ViT-B/16|-| 36.99 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_B_16_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.log) |
|  Student | ViT-T/16|Baseline|30.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.log) |
|  Student | ViT-T/16| CLIP-KD |34.90|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/ViT_T_16_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_ViT-T-16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_ViT-T-16_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |Baseline|32.60|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilevit_s_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |CLIP-KD|35.96|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/mobilevit_s_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | Swin-T |Baseline|36.38|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/swin_tiny_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | Swin-T |CLIP-KD|40.18|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/swin_tiny_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |Baseline|25.11|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilenetv3_small_100_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |CLIP-KD|26.95|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/mobilenetv3_small_100_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12m_ep32.log) |
|  Student |  EfficientNet-B0 |Baseline|32.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/efficientnet_b0_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.log) |
|  Student |  EfficientNet-B0 |CLIP-KD|35.44|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/efficientnet_b0_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.log) |
|  Student |  ResNet-18 | Baseline|28.55 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN18_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.log) |
|  Student |  ResNet-18 | CLIP-KD|31.36|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD/RN18_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-resnet18_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_cc3m_12m_kd_timm-resnet18_cc3m_12m_ep32.log) |

### Supervised by ResNet-101 as the teacher
The teacher is pretrained on CC3M+12M. Students are distilled on CC3M+12M.
| Role | Network |Method | ImageNet Acc| Train script | Download |
|:----:  | :----: | :----:  |:----:  |:----:  |:----:  |
|  Teacher |  ResNet-101 |-| 36.76 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN101_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |Baseline|32.60|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilevit_s_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | MobileViT-S |CLIP-KD|34.97|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/mobilevit_s_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-mobilevit_s_cc3m_12m_ep32.log) |
|  Student | Swin-T |Baseline|36.38|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/swin_tiny_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | Swin-T |CLIP-KD|39.51|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/swin_tiny_kd.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-swin_tiny_patch4_windows7_224_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |Baseline|25.11|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/mobilenetv3_small_100_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm_mobilenetv3_small_100_cc3m_12m_ep32.log) |
|  Student | MobileNetV3 |CLIP-KD|26.15|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/mobilenetv3_small_100_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm_mobilenetv3_small_100_cc3m_12.log) |
|  Student |  EfficientNet-B0 |Baseline|32.55|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/efficientnet_b0_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-efficientnet-b0_cc3m_12m_ep32.log) |
|  Student |  EfficientNet-B0 |CLIP-KD| 34.64|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/efficientnet_b0_kd.sh)| [model]() \| [log]() |
|  Student |  ResNet-18 | Baseline|28.55 |[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN18_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/timm-resnet18_cc3m_12m_ep32.log) |
|  Student |  ResNet-18 | CLIP-KD|30.88|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/RN101_KD/RN18_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN101_cc3m_12m_kd_timm-efficientnet-b0_cc3m_12m_ep32.log) |


### Transferred from Laion-400M
The teacher is pretrained on Laion-400M. Students are distilled on CC3M+12M.

| Role | Network | Method | ImageNet | Train script | Download |
| :----: | :----: | :----: | :----: | :----: | :----:|
|  Teacher |  ViT-L/14 |-| 72.8 |-|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_L_14-laion400m_ep32.pt)|
|  Student | ViT-B/16 |Baseline|37.0| [sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_B_16_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_cc3m_12m_ep32.log)|
|  Student | ViT-B/16 |CLIP-KD|57.5|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_L_14_KD_Laion/ViT_B_16_kd.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-B-16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-B-16_cc3m_12m_ep32.log)|
|  Student | ViT-T/16 |Baseline|30.6|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.log)|
|  Student | ViT-T/16 |CLIP-KD|40.9|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_L_14_KD_Laion/ViT_T_16_kd.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-T-16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-L-14_laion400m_kd_ViT-T-16_cc3m_12m_ep32.log)|


| Role | Network | Method | ImageNet | Train script | Download |
| :----: | :----: | :----: | :----: |:----:|:----:|
|  Teacher |  ViT-B/16 |-| 67.1 |-|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16-laion400m_e32.pt)|
|  Student | ViT-T/16 |Baseline|30.6| [sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/ViT_T_16_baseline.sh)|[model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_T_16_cc3m_12m_ep32.log)|
|  Student | ViT-T/16 |CLIP-KD|42.6|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD_Laion/ViT_T_16_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.log) |
|  Student | ResNet-50 |Baseline|35.3|[sh](https://github.com/winycg/CLIP-KD/blob/main/script/baseline/RN50_baseline.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN50_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/RN50_cc3m_12m_ep32.log) |
|  Student | ResNet-50 |CLIP-KD|55.4| [sh](https://github.com/winycg/CLIP-KD/blob/main/script/ViT_B_16_KD_Laion/RN50_kd.sh)| [model](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_laion400m_kd_RN50_cc3m_12m_ep32.pt) \| [log](https://github.com/winycg/CLIP-KD/releases/download/CLIP-KDv0.1/ViT-B-16_laion400m_kd_RN50_cc3m_12m_ep32.log) |

### Evaluate pretrained models on more downstream tasks

Evaluation a pretrained model on MSCOCO and Flickr cross-retrieval and ImageNet variants (ImageNet-V2, ImageNet-Rendition and ImageNet-Sketch) classification. Please refer to [eval_coco.sh](https://github.com/winycg/CLIP-KD/blob/main/script/eval/eval_coco.sh) and [eval_flickr.sh](https://github.com/winycg/CLIP-KD/blob/main/script/eval/eval_flickr.sh).

## Offline teacher cache and sample filtering

This repository now supports an optional offline teacher cache workflow for paired image-text training data. The cache stores normalized teacher image features, normalized teacher text features, optional candidate-caption teacher text features, and per-sample metadata with `torch.save`, so you can reuse teacher outputs across multiple distillation runs without changing the original CLIP-KD training path.

### Generate offline teacher cache

Use `src/training/cache_teacher_features.py` or the example shell script `script/cache_teacher_features.sh` to precompute teacher features.

```bash
cd src
python -m training.cache_teacher_features \
    --output-cache ../cache/teacher_features.pt \
    --teacher-model ViT-B-16 \
    --teacher-pretrained ../pretrained_models/vit_b_16.pt \
    --train-data "path/to/cc3m_train.csv,path/to/cc12m.csv" \
    --data-root "path/to/cc3m/images/,path/to/cc12m/images/" \
    --dataset-type csv \
    --csv-img-key filepath \
    --csv-caption-key title \
    --csv-caption-candidates-keys synthetic_caption,llm_caption \
    --csv-synthetic-caption-keys synthetic_caption,llm_caption \
    --csv-metadata-keys caption_quality_score,teacher_disagreement \
    --cache-density-k 32 \
    --batch-size 128 \
    --workers 8 \
    --device cuda
```

The generated cache uses the following structure:

```python
{
    "image_features": Tensor[N, D],
    "text_features": Tensor[N, D],
    "candidate_text_features": Optional[Tensor[N, K, D]],
    "candidate_text_feature_mask": Optional[BoolTensor[N, K]],
    "metadata": List[Dict],
    "config": Dict,
    "summary": Dict,
}
```

Metadata can now include:

- `sample_id`, `caption_length`, `image_text_similarity`, `difficulty_score`
- `caption_quality`
- `teacher_entropy`
- `caption_disagreement`
- `density_score`
- `candidate_caption_count`
- `synthetic_caption_count`
- per-candidate fields such as `candidate_text_similarity::<column_name>` and `caption_quality::<column_name>`

This makes the cache usable not only for feature reuse, but also for low-budget filtering, curriculum sampling, and real-vs-synthetic caption ablations.

### Train with offline feature distillation

Offline feature KD is optional and only becomes active when `--use-offline-teacher-cache` and `--teacher-cache-path` are both provided.

```bash
cd src
python -m training.main_kd \
    --train-data "path/to/cc3m_train.csv,path/to/cc12m.csv" \
    --data-root "path/to/cc3m/images/,path/to/cc12m/images/" \
    --dataset-type csv \
    --use-offline-teacher-cache \
    --teacher-cache-path ../cache/teacher_features.pt \
    --offline-kd-mode offline_only \
    --offline-kd-loss-type cosine \
    --lambda-image-kd 0.5 \
    --lambda-text-kd 0.5 \
    --offline-text-candidate-strategy softmax \
    --lambda-text-kd-candidates 0.25 \
    --offline-text-candidate-temperature 1.0
```

Supported offline KD loss arguments:

- `--use-offline-teacher-cache`
- `--teacher-cache-path`
- `--offline-kd-mode`
- `--offline-kd-loss-type` with `mse` or `cosine`
- `--lambda-image-kd`
- `--lambda-text-kd`
- `--offline-text-candidate-strategy` with `disabled`, `mean`, `max`, `softmax`, or `top1`
- `--lambda-text-kd-candidates`
- `--offline-text-candidate-temperature`

`--offline-kd-mode hybrid` is the default and preserves the original online CLIP-KD path while adding cached feature KD. `--offline-kd-mode offline_only` disables teacher forward and uses only student CLIP loss plus cached teacher feature KD.

When candidate captions are available in the cache, text KD can also distill toward an aggregated teacher text target built from synthetic or alternative captions. This keeps "real caption only" vs "real + synthetic captions" as a clean ablation variable.

When offline cache wrapping is enabled, the per-sample cache payload exposed to training uses:

```python
{
    "teacher_image_features": Tensor[D],
    "teacher_text_features": Tensor[D],
    "candidate_teacher_text_features": Optional[Tensor[K, D]],
    "candidate_teacher_text_feature_mask": Optional[BoolTensor[K]],
    "caption_length": int,
    "image_text_similarity": float,
    "difficulty_score": float,
}
```

### Train with sample filtering / curriculum sampling

Sample filtering depends on offline cache metadata. If `--use-offline-teacher-cache` is not enabled, filtering is skipped with a warning because cache metadata is unavailable.

```bash
cd src
python -m training.main_kd \
    --train-data path/to/cc3m_train.csv \
    --data-root path/to/cc3m/images \
    --dataset-type csv \
    --use-offline-teacher-cache \
    --teacher-cache-path ../cache/teacher_features.pt \
    --sample-filter-strategy curriculum_weighted \
    --sample-keep-ratio 0.6 \
    --curriculum-epoch 8 \
    --sample-weight-similarity 1.0 \
    --sample-weight-entropy 0.5 \
    --sample-weight-disagreement 0.5 \
    --sample-weight-density 0.25 \
    --sample-weight-caption-quality 0.5 \
    --offline-kd-loss-type mse \
    --lambda-image-kd 0.5 \
    --lambda-text-kd 0.5
```

Supported filtering arguments:

- `--sample-filter-strategy random`
- `--sample-filter-strategy top_similarity`
- `--sample-filter-strategy middle_similarity`
- `--sample-filter-strategy high_entropy`
- `--sample-filter-strategy high_density`
- `--sample-filter-strategy high_caption_quality`
- `--sample-filter-strategy high_disagreement`
- `--sample-filter-strategy weighted`
- `--sample-filter-strategy curriculum`
- `--sample-filter-strategy curriculum_weighted`
- `--sample-keep-ratio`
- `--curriculum-epoch`
- `--sample-weight-similarity`
- `--sample-weight-entropy`
- `--sample-weight-disagreement`
- `--sample-weight-density`
- `--sample-weight-caption-quality`
- `--sample-weight-difficulty`

Filtering behavior:

- `random`: randomly keeps `sample_keep_ratio` samples.
- `top_similarity`: keeps the highest-similarity samples according to cache metadata.
- `middle_similarity`: keeps samples from the middle similarity band and filters both very easy and very hard pairs.
- `high_entropy`: requires `teacher_entropy` in metadata; otherwise logs a warning and falls back to `random`.
- `high_density`: prefers samples from dense regions in the teacher feature space, which can be useful when you want to concentrate on representative modes.
- `high_caption_quality`: prefers samples with cleaner or richer captions according to cached heuristic quality scores.
- `high_disagreement`: prefers samples where the main caption and candidate captions disagree more strongly, which is a lightweight proxy for more informative supervision.
- `weighted`: ranks samples by a weighted combination of similarity, entropy, disagreement, density, caption quality, and difficulty.
- `curriculum`: prefers high-similarity samples in early epochs, then gradually mixes in lower-similarity or harder samples.
- `curriculum_weighted`: starts from cleaner, denser, easier examples and gradually shifts toward high-entropy, high-disagreement, or harder examples based on the configured weights.

Current support notes:

- Offline cache wrapping and sample filtering are currently supported for training datasets built through `csv`, multi-`csv`, and `synthetic`.
- `webdataset`, `vl_imagenet`, `icar`, and retrieval-style datasets are not yet supported for offline cache alignment.
- A dedicated filtered distributed sampler is included so standard distributed training is not obviously broken, but the current implementation is still designed around the supported dataset types above.


## Acknowledgement
Our codebase is bulit over [open_clip](https://github.com/mlfoundations/open_clip), an open-source codebase to run CLIP models.

We would appreciate it if our paper and repo are helpful to you!
```
@inproceedings{yang2024clip,
  title={CLIP-KD: An Empirical Study of CLIP Model Distillation},
  author={Yang, Chuanguang and An, Zhulin and Huang, Libo and Bi, Junyu and Yu, Xinqiang and Yang, Han and Diao, Boyu and Xu, Yongjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
