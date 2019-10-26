import argparse
from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()

# FB15k
parser.add_argument('--dataset', type=str, default='FB15k', help='which dataset to use')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--sample', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='hidden dimension of relation embeddings')
parser.add_argument('--iteration', type=int, default=2, help='number of GNN iterations')
parser.add_argument('--l2', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--aggregator', type=str, default='mean', help='aggregation function of GNN: mean, concat')
parser.add_argument('--feature', type=str, default='id', help='relation feature type: id, bow, bert-small, bert-large')
parser.add_argument('--finetune', type=bool, default=True, help='whether to fin tune bert features')

args = parser.parse_args()
data = load_data(args)
train(args, data)
