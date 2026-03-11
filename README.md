<div align="center">

<h1> TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations </h1>

A LLM-based Recommender System with user&item Tokenizers and a generative retrieval paradigm. The overall framework of the proposed TokenRec, which consists of the masked vector-quantized tokenizer with a K-way encoder for item ID tokenization and the generative retrieval paradigm for recommendation generation. 

<h5 align="center"> If you find this project useful, please give us a star🌟.


<h5 align="center"> 

<a href='https://arxiv.org/abs/2406.10450'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://ieeexplore.ieee.org/document/11129873'><img src='https://img.shields.io/badge/Paper-IEEE-blue'></a>

[Haohao Qu]()<sup>1</sup>,
[Zihuai Zhao]()<sup>1</sup>,
[Wenqi Fan]()<sup>1</sup>,
[Qing Li]()<sup>1</sup>,



<sup>1</sup>[The Hong Kong Polytechnic University](https://www.polyu.edu.hk/)



</h5>
</div>

## Example of Implementation


### Setup
First, please install the required dependencies using the following command:
```bash
conda create -n tokenrec
conda activate tokenrec
pip install -r requirements.txt
```
The environment includes seven packages solely: torch, torchmetrics, tqdm, transformers, pandas, numpy, and kmeans_pytorch.

[Optional] Please download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/12OFUuX7a5v7khx_MZiel04N0x5prkdGy?usp=drive_link) and place them in the "checkpoints/" path for the inference-only implementation.

### Training
Get into the "code" direction:
```bash
cd code
```

A simple example on the small dataset LastFM to run the LLM finetuning with the default configuration:
```bash
python main.py
```
To run the whole pipeline (tokenizer + backbone):
```bash
python main.py --vq --train_vq
```

For other datasets, we need to set up the correct token number and codebook number:
```bash
python main.py --dataset=ML1M --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3
python main.py --dataset=Beauty --vq --train_vq --vq_model=MQ --n_token=512 --n_book=3
python main.py --dataset=Clothing --vq --train_vq --vq_model=MQ --n_token=512 --n_book=3
```

If you want to finetune the model based on a certain checkpoint:
```bash
python main.py --dataset=LastFM --n_token=256 --n_book=3 --train_from_checkpoint
```

### Evaluation
```bash
python main.py --dataset=LastFM --no_train
```

More configurations can be found in the "./code/parse.py" file.


## Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@article{qu2025tokenrec,
  title={TokenRec: Learning to Tokenize ID for LLM-Based Generative Recommendations},
  author={Qu, Haohao and Fan, Wenqi and Zhao, Zihuai and Li, Qing},
  journal={IEEE Transactions on Knowledge \& Data Engineering},
  volume={37},
  number={10},
  pages={6216--6231},
  year={2025},
  publisher={IEEE Computer Society}
}
```
