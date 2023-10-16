DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 3 \
    --model "sage" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 50 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.00005 \
    --beta 100 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --dataset_type 'social' \


DATASET='dblp'

python e2e_main.py \
    --dataset $DATASET \
    --device 2 \
    --model "sage" \
    --dataset_type "social" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --threshold 50 \
    --epochs 300 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.00005 \
    --beta 20 \
    --train_ratio 1 \
    --aug_type 'rw' \


DATASET='lastfm'

python e2e_main.py \
    --dataset $DATASET \
    --dataset_type "recommendation" \
    --device 1 \
    --model "sage" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 25 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --alpha 0.0001 \
    --beta 20 \