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
from wenet.intern_lm.intern_lm_model import init_intern_lm_model

def load_model(model_path, configs):
    model = init_intern_lm_model(configs)
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state_dict, strict=True)
    del model_state_dict
    return model


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


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


def calc_perplexity(
    model,
    data_loader,
    output_dir,
    log_base,
    seed,
):
    set_all_random_seed(seed)
    
    f1 = os.path.join(output_dir, "utt2ppl")
    f2 = os.path.join(output_dir, "utt2ntokens")
    with torch.no_grad(), open(f1, 'w') as fout1, open(f2, 'w') as fout2:
        total_nll = 0.0
        total_ntokens = 0
        for batch in tqdm(data_loader):
            keys, text, text_lengths = batch
            text = text.to(device)
            text_lengths = text_lengths.to(device)
   
            nll, lengths = model.nll(text, text_lengths)
           
            # nll: (B, L) -> (B,)
            nll = nll.detach().cpu().numpy().sum(1)
            # lengths: (B,)
            lengths = lengths.detach().cpu().numpy()
            total_nll += nll.sum()
            total_ntokens += lengths.sum()

            for key, _nll, ntoken in zip(keys, nll, lengths):
                if log_base is None:
                    utt_ppl = np.exp(_nll / ntoken)
                else:
                    utt_ppl = log_base ** (_nll / ntoken / np.log(log_base))

                # Write PPL of each utts for debugging or analysis
                fout1.write("%s %f\n" % (key, utt_ppl))
                fout2.write("%s %f\n" % (key, ntoken))

        if log_base is None:
            ppl = np.exp(total_nll / total_ntokens)
        else:
            ppl = log_base ** (total_nll / total_ntokens / np.log(log_base))

        with open(os.path.join(output_dir, "ppl"), 'w') as fout:
            fout.write("%f\n" % ppl)
        with open(os.path.join(output_dir, "base"), 'w') as fout:
            if log_base is None:
                _log_base = np.e
            else:
                _log_base = log_base
            fout.write("%f\n" % _log_base)

        print("PPL: %f" % ppl)


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
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed"
    )
    parser.add_argument(
        "--log_base",
        type=float_or_none,
        default=None,
        help="The base of logarithm for Perplexity. "
        "If None, napier's constant is used.",
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

    calc_perplexity(
        model=model, 
        data_loader=data_loader,
        output_dir=args.output_dir, 
        log_base=args.log_base,
        seed=args.seed)
