DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "gin" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 100 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --alpha 0.00005 \
    --beta 100 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --dataset_type 'social' \


DATASET='dblp'

python e2e_main.py \
    --dataset $DATASET \
    --dataset_type "social" \
    --device 1 \
    --model "gin" \
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
    --alpha 0.0005 \
    --beta 20 \

DATASET='lastfm'

python e2e_main.py \
    --dataset $DATASET \
    --dataset_type "recommendation" \
    --device 1 \
    --model "gin" \
    --ori_lr 0.0002 \
    --aug_lr 0.0002 \
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
    --alpha 0.001 \
    --beta 20 \

