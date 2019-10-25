import argparse
import re
import numpy as np
from bert_serving.client import BertClient

# before running this script, start BERT service using the following command:
# bert-serving-start -model_dir YOUR_BERT_MODEL_DIR -mask_cls_sep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FB15k', help='which dataset to preprocess')
    parser.add_argument('-m', type=str, default='small', help='which BERT model to use: small or large')
    args = parser.parse_args()
    DATASET = args.d

    relations = []
    for line in open('../data/' + DATASET + '/relations.dict'):
        relation = line.strip().split('\t')[1]

        if DATASET == 'FB15k':
            # find all substrings only containing at least 2 letters (there is no upper-case letter in FB15k)
            # numbers, '.', '_', and '/' are removed using the following regular expression
            tokens = re.findall('[a-z]{2,}', relation)
            tokens = ' '.join(tokens)
            relations.append(tokens)


    # start BERT client and get relation embedding
    client = BertClient()
    relation_embedding = client.encode(relations)
    print(relation_embedding.shape)

    np.save('../data/' + DATASET + '/rel_emb_large.npy', relation_embedding)
