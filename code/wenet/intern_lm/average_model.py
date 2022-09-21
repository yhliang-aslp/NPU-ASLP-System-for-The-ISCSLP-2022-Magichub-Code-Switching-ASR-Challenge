# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: NPU, ASLP Group (Author: Qijie Shao)
import os
import argparse
import heapq
import glob

import yaml
import numpy as np
import torch


def read_log(train_log_path):
    loss_list = []
    with open(train_log_path, 'r') as fin:
        for line_str in fin:
            dic = {}
            epoch = int(line_str.strip().split(" ")[0].split(":")[1])
            dic["epoch"] = epoch
            loss = float(line_str.strip().split(" ")[1].split(":")[1])
            dic["loss"] = loss
            loss_list.append(dic)
    return loss_list


def select_top_n(loss_list, top_n, min_epoch, max_epoch):
    list_len = len(loss_list)
    id_start = 0
    id_end = list_len - 1
    if min_epoch > id_start:
        id_start = min_epoch
    if max_epoch < id_end:
        id_end = max_epoch
    select_list = heapq.nsmallest(top_n, loss_list[id_start:id_end+1], key=lambda x: x['loss'])
    return select_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--min_epoch',
                        default=0,
                        type=int,
                        help='min epoch used for averaging model')
    parser.add_argument('--max_epoch',
                        default=65536,  # Big enough
                        type=int,
                        help='max epoch used for averaging model')

    args = parser.parse_args()
    print(args)
    checkpoints = []
    val_scores = []
    if args.val_best:
        train_log_path = os.path.join(args.src_path, "train.log")
        loss_list = read_log(train_log_path)
        select_list = select_top_n(loss_list, args.num, args.min_epoch, args.max_epoch)
        print("\nselect model:\n" + "".join(["epoch {} loss {}\n"\
                .format(select_dic["epoch"], select_dic["loss"]) for select_dic in select_list]))
        path_list = [
            args.src_path + '/{}.pt'.format(int(select_dic["epoch"]))
            for select_dic in select_list
            ]
    else:
        path_list = glob.glob('{}/[!avg][!final]*.pt'.format(args.src_path))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-args.num:]
    # print(path_list)
    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)
