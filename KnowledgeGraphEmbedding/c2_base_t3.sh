#!/bin/bash
(
CUDA_VISIBLE_DEVICES=0 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/FB15k \
    --model RotatE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 -de --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=1 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/FB15k-237 \
    --model RotatE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 -de --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=2 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/wn18 \
    --model RotatE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 -de --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=3 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/wn18rr \
    --model RotatE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 -de --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=4 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/FB15k \
    --model TransE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=5 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/FB15k-237 \
    --model TransE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=6 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/wn18 \
    --model TransE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 --rel_batch 
)&
(
CUDA_VISIBLE_DEVICES=8 python3.5 -u codes/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path data/wn18rr \
    --model TransE \
    -n 128 -b 512 -d 400 \
    -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save kk --test_batch_size 16 --cpu_num 1 --rel_batch 
)&

