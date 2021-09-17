# PathCon

This repository is the Tensorflow implementation of PathCon ([paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467247)):
> Relational Message Passing for Knowledge Graph Completion  
Hongwei Wang, Hongyu Ren, Jure Leskovec  
In Proceedings of The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2021)

(A PyTorch implementation of PathCon can be found [here](https://github.com/hyren/PathCon))

![](https://github.com/hwwang55/PathCon/blob/master/model.png)

PathCon considers relational context and relational paths of (head, tail) pair in knowledge graphs for link prediction.
PathCon achieves substantial gains over state-of-the-art baselines.
Below is the result of Hit@1 on the test set of FB15K, FB15K-237, WN18, WN18RR, NELL995, and DDB14 datasets for relation prediction task:

| Method      | FB15K | FB15K-237 | WN18  | WN18RR | NELL995 | DDB14 |
| :---------: | :---: | :------:  | :---: | :----: | :----:  | :---: |
| TransE      | 94.0  | 94.6      | 95.5  | 66.9   | 78.1    | 94.8  |
| RotatE      | 96.7  | 95.1      | 97.9  | 73.5   | 69.1    | 93.4  |
| QuatE       | 97.2  | 95.8      | 97.5  | 76.7   | 70.6    | 92.2  |
| DRUM        | 94.5  | 90.5      | 95.6  | 77.8   | 64.0    | 93.0  |
| __PathCon__ | __97.4 (+/-0.2)__ | __96.4 (+/-0.1)__ | __98.8 (+/-0.1)__ | __95.4 (+/-0.2)__ | __84.4 (+/-0.4)__ | __96.6 (+/-0.1)__ |

For more results, please refer to the original paper.

### Files in the folder

- `data/`
  - `FB15k/`
  - `FB15k-237/`
  - `wn18/`
  - `wn18rr/`
  - `NELL995/`
  - `DDB14/`
- `src/`: implementation of PathCon

__Note__: We provide a `cache/` folder for each dataset, which caches the pre-computed relational paths for the dataset.
This folder is not required for running the code, because relational paths will be counted (and cached) when running the code if no corresponding cache file is found. 
**If you are going to run FB15K-237 with max_path_len=3, please first download and unzip ``paths_3.zip`` from [here](https://drive.google.com/file/d/1uF42OgIQY0f_G8z0Wwk90AQ_KEueqhsv/view?usp=sharing) and put all unzipped files under ``FB15k-237/cache/``** (these files cannot be uploaded to GitHub due to the limitation on file size).



### Running the code

```
$ python main.py
```
__Note__: The default dataset is set as WN18RR.
Hyper-parameter settings for other datasets are provided in  `main.py`.


### Required packages

The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.12.0
- numpy == 1.16.5
- scipy == 1.3.1
- sklearn == 0.21.3
