import argparse
import os
import pickle
from glob import glob


class Indexer:
    def __init__(self, dataset_dir_path, index_dir_path, config=None):
        self.dataset_dir_path, self.index_dir_path = dataset_dir_path, index_dir_path

        default_config = {
            'split_n_files': {'train': 500, 'dev': 0},
            'split_n_speakers': {'train': 109, 'dev': 0}
        }
        self.config = default_config if config is None else config

    def make_index(self):
        print(f'Starting to make index from {self.dataset_dir_path}.')
        file_list = self.gen_file_list(self.dataset_dir_path)
        index = self.split(file_list)

        assert len(index['train'].keys()) <= self.config['split_n_speakers']['train']

        os.makedirs(os.path.join(self.index_dir_path), exist_ok=True)
        out_path = os.path.join(self.index_dir_path, 'index.pkl')
        pickle.dump(index, open(out_path, 'wb'))
        print(f'The output file is saved to {out_path}')

    def gen_file_list(self, input_path):
        return sorted(glob(os.path.join(input_path, '*/*.wav')))

    def split(self, file_list):

        train = {}
        dev = {}

        for d in file_list:
            basename = os.path.join(*d.split('/')[-2:])
            speaker = basename.split('/')[0]
            if speaker in train.keys():
                if len(train[speaker]) < self.config['split_n_files']['train']:
                    train[speaker].append(basename)
            else:
                if len(train.keys()) < self.config['split_n_speakers']['train']:
                    train[speaker] = [basename]
                else:
                    if speaker in dev.keys():
                        if len(dev[speaker]) < self.config['split_n_files']['dev']:
                            dev[speaker].append(basename)
                    else:
                        if len(dev.keys()) < self.config['split_n_speakers']['dev']:
                            dev[speaker] = [basename]

        index = {
                'train': train,
                'dev': dev
                }

        return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indexing')

    parser.add_argument('-d', '--dataset_path', type=str)
    parser.add_argument('-s', '--save_path', type=str)

    args = parser.parse_args()

    indexer = Indexer(args.dataset_path, args.save_path)
    indexer.make_index()

