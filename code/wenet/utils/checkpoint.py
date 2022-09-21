# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

from linecache import lazycache
import logging
import os
import re

import yaml
import torch
from collections import OrderedDict


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def filter_modules(model_state_dict, modules):
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs

def load_biencoder_checkpoint(model: torch.nn.Module, path: str) -> dict:
    cn_model_path="path"
    en_model_path="path"
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading part from checkpoint %s for GPU' % path)
        checkpoint_cn = torch.load(cn_model_path)
        checkpoint_en = torch.load(en_model_path)
    else:
        logging.info('Checkpoint: loading part from checkpoint %s for CPU' % path)
        checkpoint_cn = torch.load(cn_model_path, map_location='cpu')
        checkpoint_en = torch.load(en_model_path, map_location='cpu')

    model_dict = model.state_dict()
    cn_part={}
    en_part={}
    for k, v in checkpoint_cn.items():
        if "encoders" in k:
            cn_part[k.replace("encoders", "encoders_cn")] = v
    for k, v in checkpoint_en.items():
        if "encoders" in k:
            en_part[k.replace("encoders", "encoders_en")] = v
        elif "decoders" in k:
            en_part[k] = v
    # import pdb; pdb.set_trace()
    model_dict.update(cn_part)
    model_dict.update(en_part)
    model.load_state_dict(model_dict)

def load_shared_encoder_checkpoint(model: torch.nn.Module, path: str) -> dict:
    cn_model_path="path"
    en_model_path="path"
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading part from checkpoint %s for GPU' % path)
        checkpoint_cn = torch.load(cn_model_path)
        checkpoint_en = torch.load(en_model_path)
    else:
        logging.info('Checkpoint: loading part from checkpoint %s for CPU' % path)
        checkpoint_cn = torch.load(cn_model_path, map_location='cpu')
        checkpoint_en = torch.load(en_model_path, map_location='cpu')

    model_dict = model.state_dict()
    cn_part={}
    en_part={}
    shared_part={}
    for k, v in checkpoint_cn.items():
        if "encoders" in k:
            layer = k.split('.')[2]
            if int(layer) > 19:
                cn_part[k.replace("encoders", "encoders_cn").replace(layer,str(int(layer)-20))] = v
    for k, v in checkpoint_en.items():
        if "encoders" in k:
            layer = k.split('.')[2]
            if int(layer) > 19:
                en_part[k.replace("encoders", "encoders_en").replace(layer,str(int(layer)-20))] = v
            else:
                shared_part[k.replace("encoders", "encoders_shared")] = v
        elif "decoders" in k:
            en_part[k] = v
    # import pdb; pdb.set_trace()
    model_dict.update(cn_part)
    model_dict.update(en_part)
    model_dict.update(shared_part)
    model.load_state_dict(model_dict)

def load_ltm_checkpoint(model: torch.nn.Module, path: str) -> dict:
    path="path"
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading part from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading part from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')

    model_dict = model.state_dict()
    part={}
    for k, v in checkpoint.items():
        if "decoders" in k and "src_attn" not in k:
            part[k.replace("decoder.", "intern_lm.decoder.")] = v
    model_dict.update(part)
    model.load_state_dict(model_dict)
