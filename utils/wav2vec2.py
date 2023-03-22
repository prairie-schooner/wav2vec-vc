import s3prl.hub as hub
import torch


class Wav2Vec2:
    def __init__(self, device):
        self.device = device
        self.model = getattr(hub, 'wav2vec2')().to(self.device)
        self.model.eval()

    def wav2w2v(self, wav):
        wav = [torch.tensor(wav).float().to(self.device)]
        with torch.no_grad():
            outputs = self.model(wav)
        hidden_states = [hs.cpu() for hs in outputs["hidden_states"]]
        return hidden_states
