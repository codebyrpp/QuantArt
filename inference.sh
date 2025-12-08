python inference.py \
  --content "imgs/content/6.jpg" \
  --style "imgs/style/6.jpg" \
  --config_discrete "logs/landscape2art/configs/test.yaml" \
  --config_continuous "logs/landscape2art_continuous/configs/test.yaml" \
  --ckpt_discrete "logs/landscape2art/checkpoints/last.ckpt" \
  --ckpt_continuous "logs/landscape2art_continuous/checkpoints/last.ckpt" \
  --alpha 1 \
  --beta 0.5 \
  --output "result.png"