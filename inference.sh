#!/bin/bash

python inference.py \
  --config_discrete "logs/landscape2art/configs/test.yaml" \
  --config_continuous "logs/landscape2art_continuous/configs/test.yaml" \
  --ckpt_discrete "logs/landscape2art/checkpoints/last.ckpt" \
  --ckpt_continuous "logs/landscape2art_continuous/checkpoints/last.ckpt" \