#!/bin/sh

# 1 Tasks (Offline)

# 5 Tasks
## seed1
# CUDA_VISIBLE_DEVICES=1 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 1 --n_LP_epochs 10 --dataset "cifar10" --cl_method "vncr" --buffer_capacity 500 --rep_method "supconmlp" --batch_size 128
# CUDA_VISIBLE_DEVICES=1 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 1 --n_LP_epochs 10 --dataset "cifar10" --cl_method "scr" --buffer_capacity 1000 --rep_method "supconmlp" --batch_size 128
CUDA_VISIBLE_DEVICES=1 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 1 --n_LP_epochs 10 --dataset "cifar10" --cl_method "scr" --buffer_capacity 500 --rep_method "supconmlp" --batch_size 128
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "finetuning" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "er" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "der" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "lwf" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 6 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "crl" --rep_method "supconmlp"
## seed2
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 7 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "finetuning" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 7 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "er" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 7 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "der" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 7 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "lwf" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 7 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "crl" --rep_method "supconmlp"
## seed3
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 8 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "finetuning" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 8 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "er" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 8 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "der" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 8 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "lwf" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 8 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "crl" --rep_method "supconmlp"
## seed4
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 9 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "finetuning" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 9 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "er" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 9 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "der" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 9 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "lwf" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 9 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "crl" --rep_method "supconmlp"
## seed5
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 10 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "finetuning" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 10 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "er" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 10 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "der" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 10 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "lwf" --rep_method "supconmlp"
# CUDA_VISIBLE_DEVICES=0 python test_supcon.py --seed 10 --n_tasks 5 --n_epochs 100 --dataset "cifar10" --cl_method "crl" --rep_method "supconmlp"
