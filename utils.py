import os
import pickle as pk
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def str2bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError("invalid truth value {}".format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1.0 / (1.0 + decay * step)


def set_seed(args):
    """
    set initial seed for reproduction
    """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
        torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle


def set_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.0001)
        except:
            pass
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        try:
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        except:
            pass

def load_parameters(trg_state, path):
    loaded_state = torch.load(path)
    for name, param in loaded_state.items():
        origname = name
        if name not in trg_state:
            name = name.replace("module.", "")
            name = name.replace("speaker_encoder.", "")
            if name not in trg_state:
                print("%s is not in the model."%origname)
                continue
        if trg_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, trg_state[name].size(), loaded_state[origname].size()))
            continue
        trg_state[name].copy_(param)

