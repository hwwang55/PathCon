import argparse
from data_loader import load_data
from train import train
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()

# wn18rr
parser.add_argument('--dataset', type=str, default='wn18rr', help='dataset name')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--sample', type=int, default=8, help='number of sampled neighboring edges for an entity')
parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--iteration', type=int, default=2, help='number of iterations')
parser.add_argument('--l2', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--aggregator', type=str, default='concat', help='aggregation function: mean, concat')
parser.add_argument('--feature', type=str, default='id', help='relation feature type: id, bow, bert-small, bert-large')

args = parser.parse_args()
data = load_data(args)
train(args, data)
