import argparse
import os
import torch
from torch.serialization import default_restore_location
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    df1 = pd.read_pickle(os.path.join(args.input, 'embedding.pkl'))
    state_dict = torch.load(args.model, map_location=lambda s, l: default_restore_location(s, "cpu"))
    weight = state_dict['question_encoder.encode_proj.weight'].numpy()
    bias = state_dict['question_encoder.encode_proj.bias'].numpy()
    df1['embedding'] = df1['embedding'].dot(weight.T)+bias
    df1.to_pickle(os.path.join(args.output, 'embedding.pkl'))
