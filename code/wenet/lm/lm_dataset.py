# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
# Copyright (c) 2021 Jinsong Pan
#               NPU, ASLP Group (Author: Qijie Shao)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import codecs
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from wenet.utils.common import IGNORE_ID
import yaml
# import pdb

def CollateFunc(batch):
    keys = []
    texts = []
    text_lens = []

    for data_idx in batch:
        key_idx, text_idx, text_len_idx = data_idx
        keys.append(key_idx)
        text_idx = np.array([int(x) for x in text_idx.strip().split()])
        texts.append(torch.Tensor(text_idx).long())
        text_lens.append(torch.Tensor([int(text_len_idx)]).int())
    texts = pad_sequence(texts, batch_first=True, padding_value=IGNORE_ID)
    text_lens = torch.stack(text_lens, dim=0).squeeze(1)
    return keys, texts, text_lens


class TextDataset(Dataset):
    def __init__(self,
                 data_file,
                 max_length=10240,
                 min_length=0,
                 ):
        """Dataset for loading text data.
        data_file: wenet format data file.
        """
        data = []

        # Open in utf8 mode since meet encoding problem
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    arr = line.strip().split('\t')
                    key = arr[0].split(':')[1]
                    tokenid = arr[5].split(':')[1]
                    token_length = int(arr[6].split(':')[1].split(',')[0])
                    vocab_size = int(arr[6].split(':')[1].split(',')[1])
                    data.append((key, tokenid, token_length))
                except:
                    print(line)
            self.vocab_size = vocab_size

        valid_data = []
        for i in range(len(data)):
            length = data[i][2]
            # remove too lang or too short utt for both input and output
            # to prevent from out of memory
            if length > max_length or length < min_length:
                # logging.warn('ignore utterance {} feature {}'.format(
                #     data[i][0], length))
                pass
            else:
                valid_data.append(data[i])
        self.data = valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='config file')
    parser.add_argument('--data_file', help='input data file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    dataset_conf = configs.get('dataset_conf', {})

    dataset = TextDataset(args.data_file, 
                            max_length=dataset_conf['max_length'],
                            min_length=dataset_conf['min_length'],
                            )

    data_loader = DataLoader(dataset,
                             batch_size=dataset_conf['batch_size'],
                             shuffle=True,
                             sampler=None,
                             num_workers=0,
                             collate_fn=CollateFunc,
                             )

    for i, batch in enumerate(data_loader):
        key,token,length=batch
        # pdb.set_trace()
