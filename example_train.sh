source data.sh

exp="./experiments/ec_example.yaml"
ckpt="./ckpt/ec_example/best_val_loss.pth"

$pint ec_train.py --exp $exp