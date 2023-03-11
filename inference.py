import torch
import argparse
import soundfile as sf
from models.wav2vec_vc import Wav2vecVC
from preprocess import Preprocessor


def preprocess_wav(wav_path, preprocessor, device):
    with torch.no_grad():
        w2v = preprocessor.preprocess_one((wav_path, None), None, feat='w2v')
        w2v = [layer_hs.float().to(device) for layer_hs in w2v['w2v']]
        padding_mask = torch.zeros(w2v[-1].size(0), w2v[-1].size(1)).bool().to(device)
        return w2v, padding_mask


def inference(checkpoint_path,
              vocoder_path,
              source_wav_path,
              target_wav_path,
              out_wav_path
              ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2vecVC().to(device)

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    vocoder = torch.jit.load(vocoder_path, map_location=device)
    vocoder.eval()

    preprocessor = Preprocessor(device=device)

    with torch.no_grad():
        src_w2v, src_padding_mask = preprocess_wav(source_wav_path, preprocessor, device)
        tgt_w2v, tgt_padding_mask = preprocess_wav(target_wav_path, preprocessor, device)

        print("\nConverting voice by Wav2vec-VC ...")
        dec = model.inference(src_w2v, tgt_w2v, src_padding_mask, tgt_padding_mask)
        print("Reconstructing decoded mel spectrograms ...")
        wav = vocoder.generate([dec.squeeze(0), ])[0].to('cpu')
        print("Saving wav file ...")
        sf.write(out_wav_path, wav.to('cpu'), samplerate=16000)
        print("Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-v', '--vocoder_path', type=str)
    parser.add_argument('-s', '--source_wav', type=str)
    parser.add_argument('-t', '--target_wav', type=str)
    parser.add_argument('-o', '--output_wav', type=str)

    args = parser.parse_args()

    inference(
        checkpoint_path=args.checkpoint_path,
        vocoder_path=args.vocoder_path,
        source_wav_path=args.source_wav,
        target_wav_path=args.target_wav,
        out_wav_path=args.output_wav
    )

