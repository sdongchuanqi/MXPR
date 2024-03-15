# Multimodal Cross-lingual Phrase Retrieval
This repository contains the code and pre-trained models for our paper Multimodal Cross-lingual Phrase Retrieval.

**************************** **Updates** ****************************

- 2/16 Our paper has been accepted to **LREC-COLING 2024**.

## Overview

We propose a Multimodal Cross-lingual Phrase Retrieval that extracts phrase representations from multimodal data.

![](figure/overview_xpr.png)

## Dataset

<!-- ## Anatation rules:


-->


## Getting Started
In the following sections, we describe how to use our MXPR.
### Requirements
- First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `torch==1.8.1+cu111` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.8.1` should also work. 
- Then, run the following script to fetch the repo and install the remaining dependencies.
```bash
git clone https://github.com/sdongchuanqi/MXPR.git
cd MXPR
pip install -r requirements.txt
mkdir data
mkdir model
mkdir result
```
### Dataset

Before using MXPR, please process the dataset by following the steps below.

- Download Our Dataset Here: [link]()

- Unzip our dataset and move dataset into data folder. (Make sure the path in bash file is the path of dataset)

- Get relation text from m-plug [link](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl)


### Train MXPR
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
|WikiXPR|88.63|81.44|84.53|80.18|87.32|80.83|91.00|77.62|83.94|
|MXPR|**90.22**|**84.08**|**85.95**|**80.18**|**88.18**|**83.73**|**92.11**|**80.36**|**85.75**|

 -->
