# Wav2vec-VC

This is the official implementation of the paper "Wav2vec-VC: Voice conversion via hidden representations of wav2vec 2.0".
This work utilizes all-layer hidden representations of wav2vec 2.0.
We aggregate those representations using the weighted-sum of them. 
The speaker/content layer weights are pre-trained to perform 
the speaker/content-related speech task. Given those layer weights, 
we freeze and embed them into the speaker/content encoder of Wav2vec-VC. 




## Usage

### Indexing

Before preprocessing, we can configure 1) the number of speakers 
and 2) the number of utterances per a speaker. Given such configuration, 
the train/dev set is split.

```
python index.py \
        -d dataset/VCTK/wav48 \
        -s dataset/VCTK/index
```

### Preprocessing

To feed all-layer hidden representations of wav2vec 2.0, we need to 
extract them from the pre-trained wav2vec 2.0 base model.
```
python preprocess.py \
        -d dataset/VCTK/wav48 \
        -i dataset/VCTK/index/index.pkl \
        -s dataset/VCTK/preprocessed \
        -f {w2v, mel}
```

### Training the model

Train the Wav2vec-VC model using pre-trained speaker/content layer weights.
```
python train.py \
        -f dataset/VCTK/preprocessed \
        -o checkpoints \
        -s checkpoints/speaker_layer_weight.pt \
        -c checkpoints/content_layer_weight.pt \
```

### Inference

Infer the voice-converted utterance. 
```
python inference.py \
        -c checkpoints/wav2vec-vc.pt \
        -v checkpoints/vocoder.pt \
        -s source_utterance.wav \
        -t target_utterance.wav \
        -o output_utterance.wav
```
