import argparse
import os
import torch
from torch.serialization import default_restore_location
import faiss
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--outdim', type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    shutil.copy(os.path.join(args.input, 'docid'), os.path.join(args.output, 'docid'))

    input_index = faiss.read_index(os.path.join(args.input, 'index'))
    output_index = faiss.IndexFlatIP(args.outdim)
    vectors = input_index.reconstruct_n(0, input_index.ntotal)
    print(vectors.shape)
    print('Transforming')
    state_dict = torch.load(args.model, map_location=lambda s, l: default_restore_location(s, "cpu"))
    weight = state_dict['ctx_encoder.encode_proj.weight'].numpy()
    bias = state_dict['ctx_encoder.encode_proj.bias'].numpy()
    transformed_vectors = vectors.dot(weight.T)+bias
    print(transformed_vectors)
    print('Indexing')
    output_index.add(transformed_vectors)
    faiss.write_index(output_index, os.path.join(args.output, 'index'))
