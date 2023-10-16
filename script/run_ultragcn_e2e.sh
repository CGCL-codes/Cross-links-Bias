DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "ultragcn" \
    --dataset_type "social" \
    --ori_lr 0.005 \
    --aug_lr 0.005 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 50 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --alpha 0.00005 \
    --beta 100 \
    --train_ratio 1 \
    --aug_type 'rw' \


DATASET='dblp'

python e2e_main.py \
    --dataset $DATASET \
    --device 3 \
    --model "ultragcn" \
    --dataset_type "social" \
    --ori_lr 0.01 \
    --aug_lr 0.005 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 15 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.0005 \
    --beta 10 \
    --train_ratio 1 \
    --aug_type 'rw' \


DATASET='lastfm'

python e2e_main.py \
    --dataset $DATASET \
    --dataset_type "recommendation" \
    --device 0 \
    --model "ultragcn" \
    --ori_lr 0.005 \
    --aug_lr 0.005 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 10 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --alpha 0.005 \
    --beta 20 \