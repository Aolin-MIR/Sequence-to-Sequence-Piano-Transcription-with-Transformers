from multiprocessing import process
from typing import Sequence, Tuple
import librosa
import os
import numpy as np
from mid2vacob import mid2vocab
from tqdm import tqdm
import tfrecord
#token id = time*88+note-21+1 eos=0
from torchaudio_augmentations import *
import random
import torch
from torch_pitch_shift import  pitch_shift, semitones_to_ratio, ratio_to_semitones
from torchaudio_augmentations.utils import (
    add_audio_batch_dimension,
    remove_audio_batch_dimension,
    tensor_has_valid_audio_batch_dimension,
)
from merge_tfrecord import merge_record
import torchaudio
import multiprocessing
hop_width = 256
seg_width = 63
sample_rate = 12800

class MyPitchShift(PitchShift):
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        is_batched = False
        if not tensor_has_valid_audio_batch_dimension(audio):
            audio = add_audio_batch_dimension(audio)
            is_batched = True

        fast_shift = self.draw_sample_uniform_from_fast_shifts()
        y = pitch_shift(
            input=audio,
            shift=fast_shift,
            sample_rate=self.sample_rate,
            bins_per_octave=self.bins_per_octave,
        )

        if is_batched:
            y = remove_audio_batch_dimension(y)
        return y, fast_shift



def data_aug_load(wav, num_aug=3):
    # samples, sr = librosa.load(wav, sr=sample_rate)
    # print(67,samples.shape,sr)
    samples, sr = torchaudio.load(wav)
    samples=samples[0].reshape(1,-1)
    num_samples = sr * 5
    transforms = [
        # RandomResizedCrop(n_samples=num_samples),
        # RandomApply([PolarityInversion()], p=0.8),
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

    transform = ComposeMany(transforms=transforms, num_augmented_samples=num_aug)

    transformed_audio = transform(samples)
    shifts = []
    ys = []
    for x in transformed_audio:
        x = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate)(x)
        y=x.reshape([-1]).numpy()
        shift=random.randint(-24,12)
        y=librosa.effects.pitch_shift(y, sample_rate, n_steps=shift)
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
        
    return ys,shifts
# def split_audio(samples, hop_width):
#   """Split audio into frames.不包含fft等
#     e.g.
#     >>> b=tf.signal.frame(a,5,5)
#     >>> b
#     <tf.Tensor: shape=(3, 5), dtype=int64, numpy=
#     array([[0, 1, 2, 3, 4],
#        [0, 0, 0, 0, 0], 
#        [0, 0, 0, 0, 0]])>
#     >>> a
#     array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#   """
#   return tf.signal.frame(
#       samples,
#       frame_length=hop_width,
#       frame_step=hop_width,
#       pad_end=True)


def _audio_to_frames(
    samples: Sequence[float], hop_width
) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
  """Convert audio samples to non-overlapping frames and frame times."""
  # hop_with：一个hop几个samples

  samples = np.pad(samples,
                   [0, hop_width - len(samples) % hop_width],
                   mode='constant')  # 将最后不够hopsize的部分补全,和split_audio功能重复。可删除。

  return samples


def load_audio(audio):
    samples, _ = librosa.load(audio, sr=sample_rate)

    return samples


def tokenize(midfile, audio, method='cqt',aug=True):

    vocabs, audio_splits, segs_nums=[],[],[]

    s = load_audio(audio)
    if aug:
        samples_aug,shifts=data_aug_load(audio,num_aug=1)
    else:
        samples_aug, shifts=[],[]
    samples_aug.append(s)
    shifts.append(1)

    for samples,shift in zip(samples_aug,shifts):
        frames = _audio_to_frames(samples, hop_width)
        # print (88,frames.shape)

        if frames.shape[0] < sample_rate*10:
            continue  # 过短10s以下被舍弃
        # print(183, frames.shape)
        # frames = np.reshape(frames, [-1])
        # frames = frames[:256*16]

        if method == 'cqt':
            frames = librosa.cqt(frames, sr=sample_rate,
                                hop_length=hop_width, fmin=27.50, n_bins=nbins, bins_per_octave=36)
        elif method == 'stft':
            frames = librosa.stft(frames, nfft=hop_width, hop_length=hop_width)
        elif method == 'melspec':
            frames = librosa.feature.melspectrogram(
                frames, sr=sample_rate, n_fft=2048, hop_length=hop_width, n_mels=512)
        frames = np.abs(frames)
        frames = np.transpose(frames)
        temp, nbins = frames.shape
        # print("nbins",nbins)
        frames = np.pad(frames, ((0, seg_width-temp % seg_width), (0, 0)))


        audio_split = np.reshape(frames, [-1, seg_width, nbins])
        segs_num = audio_split.shape[0]

        vocab = dump_targets(midfile, segs_num,shift)

        vocabs.append(vocab)
        audio_splits.append(audio_split)
        segs_nums.append(segs_num)

    return vocabs, audio_splits, segs_nums


def dump_targets(midfile, segs_num,shift=1):

    vocab = mid2vocab(midfile,
                      seg_width, hop_width, sample_rate, 
                      shift)
                      #int(ratio_to_semitones(shift)))
    # print(segs_num,type(segs_num))
    for _ in range(segs_num-len(vocab)):
      vocab.append([0])
    if not len(vocab)==segs_num:
        print(179,len(vocab),segs_num,shift)
    assert len(vocab) == segs_num

    return vocab


def data_maker(mid_Filelist, id,train_ds,valid_ds):
    if not id=='':
        id=str(id)
    
    writer_train = tfrecord.TFRecordWriter(train_ds+id+'.tfrecord')
    writer_valid = tfrecord.TFRecordWriter(valid_ds+id+'.tfrecord')
    train_cout, valid_cout = 0, 0

    for file in tqdm(mid_Filelist):
        # print(189)
        if random.randint(0, 79) < 1:
            aug = False
            
        else:
            aug = True

        try:
        # if True:
            vocabs, split_audios, segs_nums = tokenize(
                file, file[:-3]+'wav', method='melspec', aug=aug)
            for vocab, split_audio, segs_num in zip(vocabs, split_audios, segs_nums):

                for v, s in zip(vocab, list(split_audio)):
                    # print(178,s.shape)
                    if v == [0]:  # and random.randint(0, 100)!=5:
                        # print('ignore empty seg')

                        continue
                    if aug:
                        writer_train.write({
                            'inputs_embeds': (s.reshape([-1]).tobytes(), 'byte'),
                            'labels': (v, 'int')})
                        train_cout += 1
                        # print(train_cout)
                    else:
                        writer_valid.write({
                            'inputs_embeds': (s.reshape([-1]).tobytes(), 'byte'),
                            'labels': (v, 'int')})

                        valid_cout += 1

        except AssertionError as e:
            print('something goes wrong, but ignored.',e)
            # print(file,'too short <10s')
            continue
    writer_train.close()
    writer_valid.close()
    return (train_cout, valid_cout)

def make_datasets(path, train_ds,valid_ds,processes=10 ):
    mid_Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            if 'mid' == filename[-3:] and 'byte' not in filename:
                mid_Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    train_cout, valid_cout=0,0
    # writer_train = tfrecord.TFRecordWriter(train_ds)
    # writer_valid = tfrecord.TFRecordWriter(valid_ds)
    if processes > 1:
        temp = [[] for i in range(processes)]
        for i,x in enumerate(mid_Filelist):
            temp[i%processes].append(x)
   
        pool=multiprocessing.Pool(processes=processes)
        result=[]
        for i in range(processes):
            # print(len(temp[i]))
            result.append(pool.apply_async(data_maker,args=(
                temp[i], i, train_ds, valid_ds)))
            print('process ',i,' start.')
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()

        print('All processes done!')
        for x in result:
            temp=x.get()
            train_cout += temp[0]
            valid_cout += temp[1]
    else:
        return data_maker(mid_Filelist, '', train_ds, valid_ds)

    return train_cout,valid_cout


if __name__ == "__main__":
    import sys
    ds = sys.argv[1]
    num = 10
    if len(sys.argv) == 3:
        num = int(sys.argv[2])
    path = '/home/li/piano/workspace/midi/original'
    path = '/mnt/data/piano/workspace/midi/original'
    train_ds = ds+'_train'
    valid_ds = ds+'_valid'
    t,v = make_datasets(path, train_ds,valid_ds,processes=num)
    # from tf_record_split import split_record
    # split_record(output_file)
    # os.system("rm "+output_file)
    # from train_dev_split import merge_record
    # merge_record('aug_train.tfrecord')
    # os.system("rm /mnt/data/output_file-part-*")
    print("train", t,'valid',v)
    merge_record(train_ds, num)
    merge_record(valid_ds, num)
    # audio,shift=data_aug_load('test.wav')
    # print(len(audio),shift)