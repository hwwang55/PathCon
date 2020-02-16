# PathCon

This repository is the implementation of PathCon ([arXiv]()):
> Entity Context and Relational Paths forKnowledge Graph Completion  
Hongwei Wang, Hongyu Ren, Jure Leskovec  
arXiv Preprint

![](https://github.com/hwwang55/PathCon/blob/master/model.png)

PathCon considers relational context and relational paths of (head, tail) pair in knowledge graphs for link prediction.
PathCon achieves substantial gains over state-of-the-art baselines.
Below is the result of mean test set Hit@1 on FB15K, FB15K-237, WN18, WN18RR, NELL995, and DDB14 datasets for relation prediction task:

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
- `src/`: implementation of PathCon.




### Running the code

```
$ python main.py
```
The default dataset is set as WN18RR.
Hyper-parameter settings for other datasets are provided in  `main.py`.


### Required packages

The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.12.0
- numpy == 1.16.5
- scipy == 1.3.1
- sklearn == 0.21.3
