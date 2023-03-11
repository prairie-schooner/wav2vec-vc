from multiprocessing.pool import ThreadPool
from functools import partial
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torch
import os

from utils.ssp import SpeechSignalProcessor
from utils.wav2vec2 import Wav2Vec2


class Preprocessor:
    def __init__(self, device=None, dataset_dir_path=None, feature_dir_path=None, index_path=None):
        self.dataset_dir_path, self.feature_dir_path = dataset_dir_path, feature_dir_path
        self.ssp = SpeechSignalProcessor()
        self.device = device
        self.wav2vec = None
        self.index_path = index_path

    def preprocess_one(self, input_items, out_path, feat):
        in_path, basename = input_items

        wav = self.ssp.load_wav(in_path)

        if feat == 'mel':
            mel = self.ssp.log_mel_spectrogram(wav)
            feat = {'mel': mel}
        elif feat == 'w2v':
            self.wav2vec = Wav2Vec2(self.device) if self.wav2vec is None else self.wav2vec
            w2v = self.wav2vec.wav2w2v(wav)
            feat = {'w2v': w2v}

        if out_path is not None:
            np.save(os.path.join(out_path, f'{basename}.npy'), feat)

        return feat

    def preprocess_dataset(self, index_dict, out_path, feat, njobs):

        os.makedirs(os.path.join(out_path, feat), exist_ok=True)

        if feat == 'mel':
            task = partial(self.preprocess_one, out_path=os.path.join(out_path, feat), feat=feat)
            with ThreadPool(njobs) as pool:
                _ = list(tqdm(pool.imap(task, index_dict.items()), total=len(index_dict), desc=f'Preprocessing '))
        elif feat == 'w2v':
            for input_items in tqdm(index_dict.items()):
                self.preprocess_one(input_items, os.path.join(out_path, feat), feat=feat)

    def gen_index_dict(self, in_path, index_path):
        with open(index_path, 'rb') as f:
            index = pickle.load(f)

        file_list = []
        basename_list = []
        for dset in index.keys():
            for speaker in index[dset].keys():
                lst = [os.path.join(in_path, wav_path) for wav_path in index[dset][speaker]]
                basename_list += [os.path.basename(f) for f in lst]
                file_list += lst
        index_dict = dict(zip(file_list, basename_list))
        return index_dict

    def preprocess(self, feat, njobs):
        index_dict = self.gen_index_dict(self.dataset_dir_path, self.index_path)
        self.preprocess_dataset(index_dict=index_dict, out_path=self.feature_dir_path, feat=feat, njobs=njobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')

    parser.add_argument('-d', '--dataset_path', type=str)
    parser.add_argument('-i', '--index_path', type=str)
    parser.add_argument('-s', '--save_path', type=str)
    parser.add_argument('-f', '--feature', choices=['w2v', 'mel'])

    args = parser.parse_args()

    dataset_path = args.dataset_path
    index_path = args.index_path
    save_path = args.save_path
    feature = args.feature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = Preprocessor(device, dataset_path, save_path, index_path)
    preprocessor.preprocess(feat=args.feature, njobs=16)