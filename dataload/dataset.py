from torch.utils.data import Dataset
from glob import glob
import numpy as np
import os


class SpeechDataset(Dataset):
    def __init__(self, feature_dir, features_to_use):
        super().__init__()
        self.metadata = self.generate_metadata(feature_dir, features_to_use)
        self.features_to_use = features_to_use

    def generate_metadata(self, feature_dir, features):

        first_feature = features[0]

        feat_path_list = glob(os.path.join(feature_dir, first_feature, '*.npy'))
        basename_list = [os.path.basename(fp) for fp in feat_path_list]

        metadata = []

        for basename in basename_list:
            meta = {}
            meta['speaker'] = basename.split('_')[0]
            for feat in features:
                meta[feat] = os.path.join(feature_dir, feat, basename)
            metadata.append(meta)

        print(f"Loaded total {len(metadata)} features - What kind of features: {features}")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        data = {
            'speaker': self.metadata[index]['speaker']
        }
        for feat in self.features_to_use:
            data[feat] = np.load(self.metadata[index][feat], allow_pickle=True).item()[feat]
        return data
