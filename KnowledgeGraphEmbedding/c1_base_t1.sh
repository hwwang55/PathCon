#!/bin/bash

# Best Configuration for RotatE
#
(
bash run.sh train RotatE FB15k 1 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de --rel_batch
)&
(
bash run.sh train RotatE FB15k-237 3 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de --rel_batch
)&
(
bash run.sh train RotatE wn18 4 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de --rel_batch
)&
(
bash run.sh train RotatE wn18rr 5 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de --rel_batch
)&
# bash run.sh train RotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries --rel_batch
# bash run.sh train RotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries --rel_batch
# bash run.sh train RotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries --rel_batch
# bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de --rel_batch
#
# Best Configuration for pRotatE
#
# bash run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 --rel_batch
# bash run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 --rel_batch
# bash run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 --rel_batch
# bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 --rel_batch
# bash run.sh train pRotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries --rel_batch
# bash run.sh train pRotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries --rel_batch
# bash run.sh train pRotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries --rel_batch
#
# Best Configuration for TransE
# 
(
bash run.sh train TransE FB15k 6 0 1024 256 1000 24.0 1.0 0.0001 150000 16 --rel_batch
)&
(
bash run.sh train TransE FB15k-237 8 0 1024 256 1000 9.0 1.0 0.00005 100000 16 --rel_batch
)&