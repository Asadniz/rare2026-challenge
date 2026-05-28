echo "=== Training DINOv3 ==="
python -m training.train \
  --model_type dinov3 \
  --batch_size 8 \
  --epochs 30 \
  --learning_rate 1e-4 \
  --backbone_lr_multiplier 0.1 \
  --dinov3_unfreeze_blocks 2 \
  --loss_type cross_entropy \
  --cv_type 5fold_cv \
  --single_fold \
  --fold_number 0 \
  --data_dir data/train \
  --aug_preset top4
