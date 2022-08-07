import os.path as osp
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


def read_list(filename):
    with open(filename, "r") as fp:
        data = fp.readlines()
        data = [_l.strip() for _l in data]
    return data


class TIMIT_base(Dataset):
    def __init__(self):
        super(TIMIT_base, self).__init__()

    @staticmethod
    def preprocess(wav_data):
        norm_factor = np.abs(wav_data).max()
        wav_data = wav_data / norm_factor
        return wav_data, norm_factor

    def load_frame(self, wav_filename, offset, f_wlen):
        wav_data, fs = sf.read(wav_filename)
        # assert offset+f_wlen<=len(wav_data)
        wav_data, norm_factor = self.preprocess(wav_data)  # normlize
        offset = min(max(offset, 0), len(wav_data) - f_wlen)  # 0 <= offset <= len(wav_data) - f_wlen
        frame = wav_data[offset:offset + f_wlen]
        return frame, norm_factor


# with data augmentation now
class TIMIT_speaker_norm(TIMIT_base):
    def __init__(self, data_root, train=True, fs=16000, wlen=200, wshift=10, phoneme=False, norm_factor=False,
                 augment=True):
        super(TIMIT_speaker_norm, self).__init__()
        data_root_processed = osp.join(data_root, "processed")
        self.data_root = data_root
        self.fs, self.wlen = fs, wlen
        self.f_wlen = int(fs * wlen / 1000)
        self.f_wshift = int(fs * wshift / 1000)
        self.split = "train" if train else "test"
        self.phoneme = phoneme
        self.norm_factor = norm_factor
        self.augment = augment
        # read csv and speaker id file
        data_list_file = osp.join(data_root_processed, "{}.scp".format(self.split))
        print("load data list file from: ", data_list_file)
        self.data = read_list(data_list_file)
        self.timit_labels = np.load(osp.join(data_root_processed, "TIMIT_labels.npy"), allow_pickle=True).item()
        self.avg_frame_num = 100  # 这是什么

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = index % len(self.data)
        filename = self.data[index]
        data_wav, fs = sf.read(osp.join(self.data_root, filename), dtype='float32')
        data_wav, norm_factor = self.preprocess(data_wav)
        data_len = len(data_wav)
        # print("data len: ", data_len)
        offset = np.random.randint(0, data_len - self.f_wlen - 1)
        if self.augment and self.split == 'train':
            data_wav = np.random.uniform(1 - 0.2, 1 + 0.2) * data_wav[offset:offset + self.f_wlen]
            data_wav = np.clip(data_wav, -1, 1)
        else:
            data_wav = data_wav[offset:offset + self.f_wlen]
        speaker_id = self.timit_labels[filename]
        rtn = (data_wav, speaker_id)
        if self.phoneme:
            rtn += (None,)
        if self.norm_factor:
            rtn += (norm_factor,)
        return rtn
