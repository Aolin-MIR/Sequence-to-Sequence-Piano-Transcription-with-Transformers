from data_aug_preprocess import data_aug_load
import torchaudio,random,librosa
from torchaudio_augmentations.utils import (
    add_audio_batch_dimension,
    remove_audio_batch_dimension,
    tensor_has_valid_audio_batch_dimension,
)
from torchaudio_augmentations import *
sample_rate=12800
def data_aug_load(wav, num_aug=3):
    # samples, sr = librosa.load(wav, sr=sample_rate)
    # print(67,samples.shape,sr)
    samples, sr = torchaudio.load(wav)
    samples = samples[0].reshape(1, -1)
    num_samples = sr * 5
    transforms = [
        # RandomResizedCrop(n_samples=num_samples),
        RandomApply([PolarityInversion()], p=0.9),
        RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
        RandomApply([Gain()], p=0.2),
        # this augmentation will always be applied in this aumgentation chain!
        # HighLowPass(sample_rate=sr),
        # RandomApply([Delay(sample_rate=sr)], p=0.5),
        # PitchShift(
        #     n_samples=num_samples,
        #     sample_rate=sr
        # ),
        RandomApply([Reverb(sample_rate=sr)], p=0.3)
    ]

    transform = ComposeMany(transforms=transforms,
                            num_augmented_samples=num_aug)

    transformed_audio = transform(samples)
    shifts = []
    ys = []
    for x in transformed_audio:
        x = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate)(x)
        y = x.reshape([-1]).numpy()
        # shift = random.randint(-24, 12)
        shift=0
        y = librosa.effects.pitch_shift(y, sample_rate, n_steps=shift)
        ys.append(y)
        ##torch_shfit 可以出现无理数的半音移调，故废弃
        '''
        x, shift = MyPitchShift(
        n_samples=num_samples,
        sample_rate=sr
        )(x)

        x=torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)(x)

        ys.append(x.reshape([-1]).numpy())
        '''
        shifts.append(shift)

    return ys, shifts

from torch_pitch_shift import pitch_shift, semitones_to_ratio, ratio_to_semitones
import soundfile as sf
y,shifts=data_aug_load('test.wav')
n=1
for audio,shift in zip(y,shifts):
    print(shift)

    sf.write(str(n)+'aug.wav', audio, 12800)
    n+=1


