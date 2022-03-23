import mido
import os
def compair(mid,mid_byte):

    

if __name__ == '__main__':
    dir = '/home/li/piano/workspace/midi/original'
    Filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            if 'mid' == filename[-3:] and 'byte' not in filename:
                Filelist.append(os.path.join(home, filename))
            
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    # print(Filelist)
    for file in Filelist:
        mid = mido.MidiFile(
            file)
        mid_byte=mido.MidiFile(filename[:-8]+'byte.mid')
        compair(mid,mid_byte)
