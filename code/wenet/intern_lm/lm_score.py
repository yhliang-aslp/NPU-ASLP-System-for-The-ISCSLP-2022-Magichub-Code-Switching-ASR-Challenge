#!/usr/bin/env python3
#                NPU, ASLP Group (Author: Qijie Shao)
import argparse
import sys
import os
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import yaml
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from wenet.lm.lm_dataset import TextDataset, CollateFunc
from wenet.lm.transformer_lm import init_lm_model
import pdb

def load_model(model_path, configs):
    model = init_lm_model(configs)
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state_dict, strict=True)
    del model_state_dict
    return model

def float_or_none(value: str) -> Optional[float]:
    """float_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=float_or_none)
        >>> parser.parse_args(['--foo', '4.5'])
        Namespace(foo=4.5)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return float(value)

def lm_score(
    model,
    data_loader,
    output_dir,
):
    f1 = os.path.join(output_dir, "utt2maxscore")
    with torch.no_grad(), open(f1, 'w') as fout1:
        for batch in tqdm(data_loader):
            keys, text, text_lengths = batch
            text = text.to(device)
            text_lengths = text_lengths.to(device)
   
            scores = model.score(text, text_lengths)
            max_scores = torch.argmax(scores, dim=2).tolist()

            # pdb.set_trace()
            
            for key, max_score in zip(keys, max_scores):
                fout1.write("%s %s\n" % (key, " ".join(str(s) for s in max_score)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calc perplexity')
    parser.add_argument(
        '--test_data', 
        required=True, 
        help='test data file'
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="gpu id for this rank, -1 for cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument(
        "--train_config", 
        type=str
    )
    parser.add_argument(
        "--model_path", 
        type=str
    )
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_dataset = TextDataset(args.test_data, 
                                max_length=float('inf'),
                                min_length=0,
                                )

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=CollateFunc,
                             )

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Init asr model from configs
    model = load_model(model_path=args.model_path, 
                        configs=configs)
    model = model.to(device)
    model.eval()

    lm_score(
        model=model, 
        data_loader=data_loader,
        output_dir=args.output_dir, 
        )
