# Multimodal Cross-lingual Phrase Retrieval
This repository contains the code and pre-trained models for our paper Multimodal Cross-lingual Phrase Retrieval.

**************************** **Updates** ****************************

- 2/16 Our paper has been accepted to **Lrec-Coling2024**.

## Overview

We propose a Multimodal Cross-lingual Phrase Retrieval that extracts phrase representations from multimodal data.

![](figure/overview_xpr.png)

## Dataset



## Getting Started
In the following sections, we describe how to use our XPR.
### Requirements
- First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `torch==1.8.1+cu111` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.8.1` should also work. 
- Then, run the following script to fetch the repo and install the remaining dependencies.
```bash
git clone git@github.com:cwszz/XPR.git
cd MXPR
pip install -r requirements.txt
mkdir data
mkdir model
mkdir result
```
### Dataset

Before using XPR, please process the dataset by following the steps below.

- Download Our Dataset Here: [link]()

- Unzip our dataset and move dataset into data folder. (Make sure the path in bash file is the path of dataset)

### Checkpoint


### Train XPR
<!-- Download XLMR checkpoint from Huggingface page: [link](https://huggingface.co/xlm-roberta-base). -->
```
bash train.sh
```

### Evaluation




## References
Please cite this paper, if you found the resources in this repository useful.
<!-- Train our method:

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch  --nproc_per_node=2 \
--master_port 29501 trainMoCo.py --output_log_dir log_output --seed 42 \
--T_para 0.06 --simclr 0 --quene_length 0  --all_sentence_num 32 --train_sample_num 4 \
--dev_sample_num 32 --dev_only_q_encoder 1 --lg 'fr'
``` -->


<!-- ```
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 predict.py  --lg 'fr' --sn '32' --test_lg 'fr' \
--output_log_dir 'test_result' --simclr 0 --quene_length 0 --T_para 0.06 --seed 42 --test_dev 0 --unsupervised 0 --wolinear 0
``` -->



<!-- ## Results:


### Supervised Setting

|Model|ar-en|de-en|en-es|en-fr|en-ja|en-ko|en-ru|en-zh|avg|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CLWE|56.14|33.62|63.71|51.26|31.62|50.14|38.67|30.02|44.40|
|CLSE|20.58|18.79|36.06|26.60|16.73|24.58|21.32|17.69|22.79|
|XPR|**88.63**|**81.44**|**84.53**|**80.18**|**87.32**|**80.83**|**91.00**|**77.62**|**83.94**|

 -->
