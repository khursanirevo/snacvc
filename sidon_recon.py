import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
import torchaudio
import transformers
from huggingface_hub import hf_hub_download


fe_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="feature_extractor_cuda.pt")
decoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")

preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
    "facebook/w2v-bert-2.0"
)

# Load models globally
fe_model = torch.jit.load(fe_path, map_location='cuda').to('cuda')
decoder_model = torch.jit.load(decoder_path, map_location='cuda').to('cuda')

def denoise_speech(audio_file, fe, decoder):

    if not os.path.exists(audio_file):
        return None

    waveform, sample_rate = torchaudio.load(audio_file)

    waveform = 0.9 * (waveform / np.abs(waveform).max())

    target_n_samples = int(48_000/sample_rate * waveform.shape[-1])
    # Ensure waveform is a tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    # If stereo, convert to mono
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    # Add a batch dimension
    waveform = waveform.view(1, -1)
    wav = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=50)
    wav_16k = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)

    restoreds = []
    feature_cache = None
    wav_16k = torch.nn.functional.pad(wav_16k, (0,24000))
    for chunk in wav_16k.view(-1).split(16000 * 60):
        inputs = preprocessor(
            torch.nn.functional.pad(chunk, (40, 40)), sampling_rate=16_000, return_tensors="pt"
        ).to('cuda')
        with torch.inference_mode():
            feature = fe(inputs["input_features"].to("cuda"))["last_hidden_state"]
            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
                restored_wav = decoder(feature.transpose(1, 2))
                restored_wav = restored_wav[:, :, 4800:]
            else:
                restored_wav = decoder(feature.transpose(1, 2))
                restored_wav = restored_wav[:, :, 50 * 3 :]
            feature_cache = feature[:, -5:, :]
        restoreds.append(restored_wav.cpu())
    restored_wav = torch.cat(restoreds, dim=-1)

    return 48_000, (restored_wav.view(-1, 1).numpy() * 32767).astype(np.int16)[:target_n_samples].T


if __name__ == "__main__":
    new_path_cleaned = "/ml-data/sabil/check_orpheus_data/Voice_Actor_reconstructed"
    os.makedirs(new_path_cleaned, exist_ok=True)

    for audio_path in tqdm(os.listdir(base_folder_path)):
        filename = os.path.join(base_folder_path, audio_path)
        sr, y_cons_int16pcm = denoise_speech(filename)
        sf.write(os.path.join(new_path_cleaned, audio_path), y_cons_int16pcm.T, samplerate=sr, subtype='PCM_16')

    # for ipynb usage
    # from IPython.display import Audio, display
    # display(Audio(filename))
    # display(Audio(y_cons_int16pcm, rate=sr))