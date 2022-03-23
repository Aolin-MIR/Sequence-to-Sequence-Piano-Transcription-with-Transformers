from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar, MutableMapping
import librosa
import os
import numpy as np
from scipy.fftpack.basic import fft
# import tensorflow as tf
# import note_seq
import dataclasses
# import event_codec
# import note_sequences
# import vocabularies
import math
from mid2vacob import mid2vocab
from tqdm import tqdm
import tfrecord
# import seqio
# from mt3 import event_codec, run_length_encoding, note_sequences, vocabularies
# from mt3.vocabularies import build_codec
import random
#token id = time*88+note-21+1 eos=0

hop_width = 128
seg_width = 127
sample_rate = 12800






@dataclasses.dataclass
class NoteEncodingState:
  """Encoding state for note transcription, keeping track of active pitches."""
  # velocity bin for active pitches and programs
  active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(
      default_factory=dict)


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








def tokenize(midfile, audio,method='cqt'):

    # note_sequences.validate_note_sequence(ns)
    samples = load_audio(audio)
    frames = _audio_to_frames(samples,hop_width)
    # print (88,frames.shape)
    assert frames.shape[0] >= sample_rate*10  # 过短10s以下被舍弃
    # print(183, frames.shape)
    # frames = np.reshape(frames, [-1])
    # frames = frames[:256*16]
    if method=='cqt':
        frames = librosa.cqt(frames, sr=sample_rate,
                         hop_length=hop_width, fmin=27.50, n_bins=nbins, bins_per_octave=36)
    elif method == 'stft':
        frames = librosa.stft(frames,nfft=hop_width, hop_length=hop_width)
    elif method == 'melspec':
        frames = librosa.feature.melspectrogram(frames, sr=sample_rate, n_fft=2048, hop_length=256, n_mels=512)
    frames = np.abs(frames)
    frames = np.transpose(frames)
    temp, nbins = frames.shape
    # print("nbins",nbins)
    frames = np.pad(frames, ((0, seg_width-temp % seg_width), (0, 0)))
    # print(191,frames.shape)
    # if onsets_only:
    #   times, values = note_sequence_to_onsets(ns)
    # else:
    #   ns = note_seq.apply_sustain_control_changes(ns)
    #   times, values = (note_sequence_to_onsets_and_offsets_and_programs(ns))

    audio_split = np.reshape(frames, [-1, seg_width, nbins])
    segs_num = audio_split.shape[0]

    vocab = dump_targets(midfile, segs_num)

    return vocab, audio_split, segs_num


def dump_targets(midfile, segs_num):

    vocab = mid2vocab(midfile,
                      seg_width, hop_width,sample_rate)
    # print(segs_num,type(segs_num))
    for _ in range(segs_num-len(vocab)):
      vocab.append([0])
    
    assert len(vocab) == segs_num

    return vocab
   


# def _float_feature(value):
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# # 获取整型数据
# def _int64_feature(value):
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# def _bytes_feature(value):
    
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



# def data_example(v, s):
#     # print(148,s.shape,s)
#     feature = {
#         'inputs_embeds': _bytes_feature(s.tostring()),  # 被用字符串保存

#         'labels': _int64_feature(v),

#     }
#     # print (154,feature)
#     return tf.train.Example(features=tf.train.Features(feature=feature))


def make_datasets(path, output_file):
    mid_Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            if 'mid' == filename[-3:] and 'byte' not in filename:
                mid_Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

    writer=tfrecord.TFRecordWriter(output_file)
    cout=0


    for file in mid_Filelist:

        try:
            vocab, split_audio, segs_num = tokenize(file, file[:-3]+'wav',method='melspec')
            for v, s in zip(vocab, list(split_audio)):
                # print(178,s.shape)
                if v == [0]: #and random.randint(0, 100)!=5:  
                    # print('ignore empty seg')  
                             
                    continue
                # if not s.shape ==(seg_width,nbins):
                #     # print(187,s.shape)
                #     continue
                # print(186,s.shape)
                writer.write({
                    'inputs_embeds': (s.reshape([-1]).tobytes(), 'byte'),
                    'labels':(v,'int')
                })
                cout+=1

        except AssertionError as e: 
            # print(file,'too short <10s') 
            continue
    writer.close()

    return cout

if __name__ == "__main__":
    # path = '/home/li/piano/workspace/midi/original/beijing/ipadx-210826-2/'
    # midifile = path+ '21-08-26-13-23-08.mid'
    # audio = path+'21-08-26-13-23-08.wav'
    # # ns = midi2noteseq(midifile)
    # # ds=tokenize(ns,audio)
    # # v, s = tokenize(midifile, audio)
    path = '/home/li/piano/workspace/midi/original'
    # path = '/mnt/data/piano/workspace/midi/original'
    output_file = 'ours_0.01_127_melspec_12800_train.tfrecord'
    cout = make_datasets(path, output_file)
    from tf_record_split import split_record
    split_record(output_file)
    os.system("rm /home/li/mt3/"+output_file)
    from train_dev_split import merge_record
    merge_record('ours_train.tfrecord')
    os.system("rm /home/li/mt3/output_file-part-*")
    print("cout", cout)
