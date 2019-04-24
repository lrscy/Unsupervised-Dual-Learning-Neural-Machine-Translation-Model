#!/bin/bash

nmtdir=nmt/models
lmdir=lm/models
srcdir=data/new

nmtA=$nmtdir/model.zhen.bin
nmtB=$nmtdir/model.enzh.bin
lmA=$lmdir/chinese.pt
lmB=$lmdir/english.pt
lmA_dict=$lmdir/dict.zh.pkl
lmB_dict=$lmdir/dict.en.pkl
srcA=$srcdir/chinese/train.txt
srcB=$srcdir/english/train.txt

saveA="modelA"
saveB="modelB"

python3 dual.py \
    --nmt $nmtA $nmtB \
    --lm $lmA $lmB \
    --dict $lmA_dict $lmB_dict \
    --src $srcA $srcB \
    --log_every 5 \
    --save_n_iter 400 \
    --alpha 0.01 \
    --model $saveA $saveB

