from transformers import Wav2Vec2Model
import torch

# if you want to obtain wav2vec 2.0 representation from S3PRL
# import s3prl.hub as hub
#
# class Wav2Vec2:
#     def __init__(self, device):
#         self.device = device
#         self.models = getattr(hub, 'wav2vec2')().to(self.device)
#
#     def wav2w2v(self, wav):
#         wav = [torch.tensor(wav).float().to(self.device)]
#         with torch.no_grad():
#             outputs = self.models(wav)
#         hidden_states = [hs.cpu() for hs in outputs["hidden_states"]]
#         return hidden_states


class Wav2Vec2:
    def __init__(self, device):
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.model.eval()

    def wav2w2v(self, wav):
        wav = torch.Tensor(wav).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values=wav,
                                 output_hidden_states=True)
        hidden_states = [hs.cpu() for hs in outputs.hidden_states]
        return hidden_states
