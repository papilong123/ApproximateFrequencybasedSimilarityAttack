import argparse
import configparser
import os
import os.path as osp
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from ApproximateFrequencySimilarityAttack.checkpoints.model import SincClassifier
from ApproximateFrequencySimilarityAttack.utils.datasets import TIMIT_speaker_norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dict_from_args(keys, args):
    data = {}
    for key in keys:
        data[key] = getattr(args, key)
    return data


def get_pretrained_model(args):
    args_all = {'speaker': args}
    models = {}
    for key, args in args_all.items():
        CNN_arch = get_dict_from_args(['cnn_input_dim', 'cnn_N_filt', 'cnn_len_filt', 'cnn_max_pool_len',
                                       'cnn_use_laynorm_inp', 'cnn_use_batchnorm_inp', 'cnn_use_laynorm',
                                       'cnn_use_batchnorm', 'cnn_act', 'cnn_drop'], args.cnn)

        DNN_arch = get_dict_from_args(['fc_input_dim', 'fc_lay', 'fc_drop',
                                       'fc_use_batchnorm', 'fc_use_laynorm', 'fc_use_laynorm_inp',
                                       'fc_use_batchnorm_inp', 'fc_act'], args.dnn)

        Classifier = get_dict_from_args(['fc_input_dim', 'fc_lay', 'fc_drop',
                                         'fc_use_batchnorm', 'fc_use_laynorm', 'fc_use_laynorm_inp',
                                         'fc_use_batchnorm_inp',
                                         'fc_act'], args.classifier)

        CNN_arch['fs'] = args.windowing.fs
        model = SincClassifier(CNN_arch, DNN_arch, Classifier)
        if args.speaker_model != 'none':
            print("load model from:", args.speaker_model)
            if os.path.splitext(args.speaker_model)[1] == '.pkl':
                checkpoint_load = torch.load(args.speaker_model, map_location=device)
                model.load_raw_state_dict(checkpoint_load)
            else:
                load_checkpoint(model, args.speaker_model, strict=True)

        model = model.to(device).eval()
        # freeze the model
        for p in model.parameters():
            p.requires_grad = False
        models[key] = model

    return models['speaker']


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url

    if not osp.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict)
    else:
        model.load_state_dict(state_dict, strict)
    return checkpoint


def _init_fn(work_id, seed):
    np.random.seed(work_id + seed)


def load_audio(opt):
    train_dataset = TIMIT_speaker_norm(opt.data_root, train=True, wlen=opt.wlen, phoneme=False, norm_factor=False,
                                       augment=False)
    train_dataloader = DataLoader(train_dataset, opt.optimization.batch_size, num_workers=opt.num_workers,
                                  pin_memory=True, shuffle=True,
                                  worker_init_fn=partial(_init_fn, seed=opt.optimization.seed))
    test_dataset = TIMIT_speaker_norm(opt.data_root, train=False, wlen=opt.wlen, phoneme=False, norm_factor=False,
                                      augment=False)
    test_dataloader = DataLoader(test_dataset, opt.optimization.batch_size, num_workers=opt.num_workers,
                                 pin_memory=True, shuffle=False,
                                 worker_init_fn=partial(_init_fn, seed=opt.optimization.seed))
    return train_dataloader, test_dataloader, len(train_dataset)


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


def read_conf(speaker_cfg, options):
    cfg_file = speaker_cfg
    print("read config file: ", cfg_file)
    Config = configparser.ConfigParser()
    Config.read(cfg_file)

    # [windowing]
    options.windowing = argparse.Namespace()
    options.windowing.fs = int(Config.get('windowing', 'fs'))
    options.windowing.cw_len = int(Config.get('windowing', 'cw_len'))
    options.windowing.cw_shift = int(Config.get('windowing', 'cw_shift'))

    # [cnn]
    options.cnn = argparse.Namespace()
    options.cnn.cnn_input_dim = int(Config.get('cnn', 'cnn_input_dim'))
    options.cnn.cnn_N_filt = list(map(int, Config.get('cnn', 'cnn_N_filt').split(',')))
    options.cnn.cnn_len_filt = list(map(int, Config.get('cnn', 'cnn_len_filt').split(',')))
    options.cnn.cnn_max_pool_len = list(map(int, Config.get('cnn', 'cnn_max_pool_len').split(',')))
    options.cnn.cnn_use_laynorm_inp = str_to_bool(Config.get('cnn', 'cnn_use_laynorm_inp'))
    options.cnn.cnn_use_batchnorm_inp = str_to_bool(Config.get('cnn', 'cnn_use_batchnorm_inp'))
    options.cnn.cnn_use_laynorm = list(map(str_to_bool, Config.get('cnn', 'cnn_use_laynorm').split(',')))
    options.cnn.cnn_use_batchnorm = list(map(str_to_bool, Config.get('cnn', 'cnn_use_batchnorm').split(',')))
    options.cnn.cnn_act = list(map(str, Config.get('cnn', 'cnn_act').split(',')))
    options.cnn.cnn_drop = list(map(float, Config.get('cnn', 'cnn_drop').split(',')))
    options.cnn.arch_lr = float(Config.get('cnn', 'arch_lr'))
    options.cnn.arch_opt = str(Config.get('cnn', 'arch_opt'))
    options.cnn.arch_opt_alpha = float(Config.get('cnn', 'arch_opt_alpha'))
    options.cnn.lr_decay_step = int(Config.get('cnn', 'lr_decay_step'))
    options.cnn.lr_decay_factor = float(Config.get('cnn', 'lr_decay_factor'))

    # [dnn]
    options.dnn = argparse.Namespace()
    options.dnn.fc_input_dim = int(Config.get('dnn', 'fc_input_dim'))
    options.dnn.fc_lay = list(map(int, Config.get('dnn', 'fc_lay').split(',')))
    options.dnn.fc_drop = list(map(float, Config.get('dnn', 'fc_drop').split(',')))
    options.dnn.fc_use_laynorm_inp = str_to_bool(Config.get('dnn', 'fc_use_laynorm_inp'))
    options.dnn.fc_use_batchnorm_inp = str_to_bool(Config.get('dnn', 'fc_use_batchnorm_inp'))
    options.dnn.fc_use_batchnorm = list(map(str_to_bool, Config.get('dnn', 'fc_use_batchnorm').split(',')))
    options.dnn.fc_use_laynorm = list(map(str_to_bool, Config.get('dnn', 'fc_use_laynorm').split(',')))
    options.dnn.fc_act = list(map(str, Config.get('dnn', 'fc_act').split(',')))
    options.dnn.arch_lr = float(Config.get('dnn', 'arch_lr'))
    options.dnn.arch_opt = str(Config.get('dnn', 'arch_opt'))
    options.dnn.lr_decay_step = int(Config.get('dnn', 'lr_decay_step'))
    options.dnn.lr_decay_factor = float(Config.get('dnn', 'lr_decay_factor'))

    # [class]
    options.classifier = argparse.Namespace()
    options.classifier.fc_input_dim = int(Config.get('classifier', 'fc_input_dim'))
    options.classifier.fc_lay = list(map(int, Config.get('classifier', 'fc_lay').split(',')))
    options.classifier.fc_drop = list(map(float, Config.get('classifier', 'fc_drop').split(',')))
    options.classifier.fc_use_laynorm_inp = str_to_bool(Config.get('classifier', 'fc_use_laynorm_inp'))
    options.classifier.fc_use_batchnorm_inp = str_to_bool(Config.get('classifier', 'fc_use_batchnorm_inp'))
    options.classifier.fc_use_batchnorm = list(
        map(str_to_bool, Config.get('classifier', 'fc_use_batchnorm').split(',')))
    options.classifier.fc_use_laynorm = list(map(str_to_bool, Config.get('classifier', 'fc_use_laynorm').split(',')))
    options.classifier.fc_act = list(map(str, Config.get('classifier', 'fc_act').split(',')))
    options.classifier.arch_lr = float(Config.get('classifier', 'arch_lr'))
    options.classifier.arch_opt = str(Config.get('classifier', 'arch_opt'))
    options.classifier.lr_decay_step = int(Config.get('classifier', 'lr_decay_step'))
    options.classifier.lr_decay_factor = float(Config.get('classifier', 'lr_decay_factor'))

    # [optimization]
    options.optimization = argparse.Namespace()
    options.optimization.lr = float(Config.get('optimization', 'lr'))
    options.optimization.batch_size = int(Config.get('optimization', 'batch_size'))
    options.optimization.N_epochs = int(Config.get('optimization', 'N_epochs'))
    options.optimization.N_eval_epoch = int(Config.get('optimization', 'N_eval_epoch'))
    options.optimization.print_every = int(Config.get('optimization', 'print_every'))
    options.optimization.seed = int(Config.get('optimization', 'seed'))

    return options
