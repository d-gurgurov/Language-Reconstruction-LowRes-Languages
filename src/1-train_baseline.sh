#!/bin/bash

pip install -r requirements.txt

fairseq-preprocess \
    --source-lang mt --target-lang en \
    --trainpref /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-train.bpe \
    --validpref /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-dev.bpe \
    --testpref /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-test.bpe \
    --destdir /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/mt-en \
    --joined-dictionary \
    --workers 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fairseq-train \
    /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/mt-en \
    --fp16 \
    --arch transformer \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --clip-norm 0.0 \
    --lr 1e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-update 150000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 10 \
    --encoder-layers 6 \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 \
    --decoder-layers 6 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 2048 \
    --decoder-attention-heads 8 \
    --save-dir checkpoints/transformer_mt_en \
    --log-format json \
    --log-interval 100 \
    --log-file checkpoints/transformer_mt_en/log_mt_en.json \
    --save-interval-updates 1000 \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --ddp-backend no_c10d \
    --distributed-world-size 4 \
    --distributed-num-procs 4 \
    --update-freq 8