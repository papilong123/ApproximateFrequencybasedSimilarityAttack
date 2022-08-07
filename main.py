import argparse
import time
from copy import deepcopy

import torch
import tqdm

from audio_cov.attack.ours import OURS
from audio_cov.utils.auxiliary_utils import setup_seed, common, predict, attack_success
from audio_cov.utils.eval_metric_utils import LpDistance
from audio_cov.utils.utils import get_pretrained_model, load_audio, read_conf


def parse_arg():
    parser = argparse.ArgumentParser(description='Approximate frequency based similarity attack')
    parser.add_argument('--bs', type=int, default=120, help="batch size")
    parser.add_argument('--classifier', type=str, default='SincNet', help='model to attack')
    parser.add_argument('--seed', type=int, default=18, help='random seed')
    parser.add_argument('--attack_method', type=str, default='ours', help='attack method')
    parser.add_argument('--wave_name', type=str, default='db6', choices=['db6', 'bior6.8'])

    # taining parameters
    parser.add_argument('--steps', type=int, default=150, help='max number steps')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='hyper parameter for adv cost')
    parser.add_argument('--beta', type=float, default=0.1, help='hyper parameter for approximate frequency constraint')
    parser.add_argument('--decomposition_level', type=int, default=1, help='how many layers to decompose')
    parser.add_argument('--kappa', type=int, default=0, help='control desired confidence')

    parser.add_argument('--data_root', type=str, default='./data/TIMIT/TIMIT_lower', help='path for data')
    parser.add_argument('--dataset', choices=['timit', 'libri'], default='timit', help='dataset to attack')
    parser.add_argument('--speaker_cfg', type=str, default='./config/timit_speaker.cfg', help='max number iteration')
    parser.add_argument('--speaker_model', type=str, default='./output/SincNet_TIMIT/model_raw.pkl',
                        help='victim model')
    parser.add_argument('--wlen', type=int, default=200, help='length for a frame data, ms')

    args = parser.parse_args()
    speaker_cfg = args.speaker_cfg
    args_speaker = read_conf(speaker_cfg, deepcopy(args))
    return args_speaker


opt = parse_arg()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if opt.seed != -1:
    setup_seed(opt.seed)

# load model and dataset
classifier = get_pretrained_model(opt)
train_dataloader, test_dataloader, num_audios = load_audio(opt)
print("Attack Benign Audio of {} dataset({} audios) with perturb mode: {} :".format(
    opt.dataset, num_audios, opt.attack_method
))


l2 = 0
l_inf = 0
lowFre = 0
total_img = 0
att_suc_img = 0

att = OURS(model=classifier,
           steps=opt.steps,
           learning_rate=opt.learning_rate,
           targeted=False,
           alpha=opt.alpha,
           beta=opt.beta,
           wave_name=opt.wave_name)

batch_idx = 0
start = time.perf_counter()
for inputs, labels in tqdm.tqdm(train_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    common_id = common(labels, predict(classifier, inputs))
    total_img += len(common_id)
    inputs = inputs[common_id].to(device)
    labels = labels[common_id].to(device)

    # attack and calculate ASR
    adv = att(inputs, labels)

    att_suc_id = attack_success(labels, predict(classifier, adv))
    att_suc_img += len(att_suc_id)

    adv = adv[att_suc_id]
    inputs = inputs[att_suc_id]

    lp = LpDistance(inputs, adv)
    l2 += lp.Lp2()
    l_inf += lp.Lpinf()
    lowFre += lp.LowFreNorm()

print("Evaluating Adversarial images of {} dataset({} images) with perturb mode:{} :".format(
    opt.dataset, total_img, opt.attack_method))
print("BatchIdx={:<5} "
      "Fooling Rate: {:.2f}% "
      "L2 Norm: {:.4f} "
      "L_inf Norm: {:.4f} "
      "Low Frequency Norm: {:.4f} "
      "Time used: {:.2f}".format(batch_idx, 100.0 * att_suc_img / total_img, l2 / att_suc_img,
                                 l_inf / att_suc_img, lowFre / att_suc_img, time.perf_counter() - start))
