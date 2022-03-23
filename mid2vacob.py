import mido
import os
# mid=mido.MidiFile('/Users/liaolin/Desktop/event2midi/21-09-01-10-23-26.mid')
# for i, track in enumerate(mid.tracks):#enumerate()：创建索引序列，索引初始为0
#     print('Track {}: {}'.format(i, track.name))
#     for msg in track:#每个音轨的消息遍历
#         print(msg)
def mid2vocab(path,seg_width,hop_width,sample_rate,shift=0):
    # seg_len*=1000
    hop_size=hop_width/sample_rate
    seg_len=hop_size*seg_width
    # print(12,seg_len)
    mid = mido.MidiFile(
        path)
    tempo=0
    for i, track in enumerate(mid.tracks):
        # print('Track {}: {}'.format(i, track.name))
        passed_time = 0

        for msg in track:
            ab_time = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            # real_time就是每一个事件在整个midi文件中的真实时间位置
            real_time = ab_time + passed_time
            passed_time += ab_time
            # print(msg)
            if msg.type== 'set_tempo':
                tempo = msg.tempo
                # print(tempo)
                # break
        # else:
        #     continue
        # break
    #segament length = 4.088s->310ms
    #hopsize = 10ms 

    vocab=[]
    seg=[]
    for i, track in enumerate(mid.tracks):
        # print('Track {}: {}'.format(i, track.name))
        passed_time = 0
        num=0
        for msg in track:
            
            ab_time = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            # real_time就是每一个事件在整个midi文件中的真实时间位置
            real_time = ab_time + passed_time
            passed_time += ab_time
            # if msg.type== 'set_tempo':
            #     print(msg)
            if msg.type == "note_on" and 21<=(msg.note+shift) <= 108 and msg.velocity!=0:
                # print(msg, " real time=" + str(real_time))
                index=real_time//seg_len
                while index >num:
                    seg.append(0)
                    vocab.append(seg)
                    seg=[]
                    num+=1
                # print(55,real_time)
                rel_time=int((real_time/hop_size)%(seg_len/hop_size))
                token_id = rel_time*88+msg.note+shift-21+1
                if token_id>88*seg_width:
                    print('有问题：',token_id, real_time,rel_time, msg.note)
                    continue
                # print(rel_time == (rel_time*88+msg.note-21+1-1) //88, (rel_time*88+msg.note-21+1-1)%88==(msg.note-21),rel_time,real_time)
                seg.append(token_id)
    seg.append(0)
    vocab.append(seg)
    # with open (path[:-3]+'vocab','w') as f:
    #     f.writelines(str(vocab))
    return vocab
if __name__ == '__main__':
    dir = '/home/li/piano/workspace/midi/original'
    dir = 'pianoIPhoneX-211026-英昌'
    Filelist=[]
    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            if 'mid'==filename[-3:] and 'byte' not in filename:
                Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    # print(Filelist)
    for file in Filelist[:1]:
        # print(vocab)
        print(mid2vocab(file,15,256,12800))
