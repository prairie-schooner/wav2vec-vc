from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(batch):

    speaker = [bc['speaker'] for bc in batch]

    mel = [torch.Tensor(bc['mel']) for bc in batch]
    mel_lens = [m.size(0) for m in mel]

    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    padding_mask = torch.stack([torch.arange(mel[-1].size(0)) >= length for length in mel_lens])

    w2v = [[] for _ in range(len(batch[0]['w2v']))]

    for bc in batch:
        for i, hs in enumerate(bc['w2v']):
            w2v[i].append(hs.squeeze(0))

    w2v = [pad_sequence(lw_w2v, batch_first=True).contiguous() for lw_w2v in w2v]

    return {'speaker': speaker,
            'mel': mel.contiguous(),
            'w2v': w2v,
            'padding_mask': padding_mask
            }

