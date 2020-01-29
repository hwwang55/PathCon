import argparse
import re
import numpy as np
from bert_serving.client import BertClient

# before running this script, start BERT service using the following command:
# bert-serving-start -model_dir YOUR_BERT_MODEL_DIR -mask_cls_sep
# download a pre-trained bert model from the following links:
# https://bert-as-service.readthedocs.io/en/latest/section/get-start.html#download-a-pre-trained-bert-model
# we use the 'BERT-Base, Uncased' (uncased_L-12_H-768_A-12) model for this repo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='wn18rr', help='which dataset to preprocess')
    args = parser.parse_args()
    dataset = args.d

    relations = []
    for line in open('../data/' + dataset + '/relations.dict'):
        relation = line.strip().split('\t')[1]

        # find all substrings only containing at least 2 letters
        tokens = re.findall('[a-z]{2,}', relation)
        tokens = ' '.join(tokens)
        relations.append(tokens)

    # start BERT client and get relation embedding
    client = BertClient()
    relation_embedding = client.encode(relations)
    print(relation_embedding.shape)

    np.save('../data/' + dataset + '/bert.npy', relation_embedding)
