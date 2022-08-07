import random

import numpy as np
import torch


def setup_seed(seed):
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        pred = outputs.max(1, keepdim=False)[1]
        return pred


def common(targets, pred):
    common_id = np.where(targets.cpu() == pred.cpu())[0]
    return common_id


def attack_success(targets, pred):
    attack_id = np.where(targets.cpu() != pred.cpu())[0]
    return attack_id
