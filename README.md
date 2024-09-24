Please download the checkpoints at [Google Drive](https://drive.google.com/drive/folders/12OFUuX7a5v7khx_MZiel04N0x5prkdGy?usp=drive_link), and put them in the path of "checkpoints/".

## An example of Implementation

1. Go to the path of "code"
```
python cd code
```

2. Evaluation
```
python main.py --dataset=LastFM --no_train
```

3. Train from scratch (MQ Tokenizers + LLM)
```
python main.py --dataset=LastFM --train_vq --vq_model=MQ --n_token=256 --n_book=3
```

3. Train from checkpoint (LLM)
```
python main.py --dataset=LastFM --train_vq --vq_model=MQ --n_token=256 --n_book=3 --train_from_checkpoint
```
