# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
#                    NPU, ASLP Group (Author: Qijie Shao)
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

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from wenet.lm.lm_dataset import TextDataset, CollateFunc
from wenet.lm.executor import Executor
from wenet.utils.scheduler import WarmupLR
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from wenet.lm.transformer_lm import init_lm_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--init_model', help='asr init model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)
    print(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    distributed = args.world_size > 1

    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = TextDataset(args.train_data, 
                                max_length=dataset_conf['max_length'],
                                min_length=dataset_conf['min_length']
                                )
    cv_dataset = TextDataset(args.cv_data, 
                                max_length=dataset_conf['max_length'],
                                min_length=dataset_conf['min_length']
                                )
    
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_data_loader = DataLoader(train_dataset,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   pin_memory=args.pin_memory,
                                   batch_size=dataset_conf['batch_size'],
                                   num_workers=args.num_workers,
                                   collate_fn=CollateFunc,
                                   )
    cv_data_loader = DataLoader(cv_dataset,
                                shuffle=False,
                                batch_size=dataset_conf['batch_size'],
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                collate_fn=CollateFunc,
                                )

    vocab_size = train_dataset.vocab_size
    # Save configs to model_dir/train.yaml for inference and export
    configs['vocab_size'] = vocab_size

    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    ################ model #####################
    # Init asr model from configs
    model = init_lm_model(configs)
    if args.rank == 0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print('the number of model params: {}'.format(num_params))

    # If specify init_model, load some info from init_model
    if args.init_model is not None:
        model_state_dict = torch.load(args.init_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state_dict, strict=False)
        del model_state_dict
        print("load model from {}".format(args.init_model))

    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    step = infos.get('step', -1)

    ############## executor ####################
    executor = Executor()
    num_epochs = configs['max_epoch']
    model_dir = args.model_dir
     
    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam([{'params':model.parameters(), 'initial_lr':configs['optim_conf']['lr']}], **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    final_epoch = None
    configs['rank'] = args.rank
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        print('Epoch {} TRAIN info lr {}'.format(epoch, lr))

        executor.train(
                model, 
                optimizer, 
                scheduler, 
                train_data_loader, 
                device,
                configs,
                scaler)
        
        if args.rank == 0:
            loss = executor.cv(model, 
                                cv_data_loader, 
                                device,
                                )
            log_str = "Epoch:%d CV_Loss:%.4f" % (epoch, loss)
            print(log_str)
            
            with open(os.path.join(args.model_dir, 'train.log'), 'a+') as flog:
                flog.write(log_str + '\n')
    
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                    model, save_model_path, {
                        'epoch': epoch,
                        'lr': lr,
                        'cv_loss': loss,
                        'step': executor.step
                    })