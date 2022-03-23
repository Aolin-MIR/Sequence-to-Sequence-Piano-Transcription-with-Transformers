import tensorflow as tf
import sys
import os
def merge_record(outpath, num=10):
    list_of_tfrecord_files = []
    for x in range(num-1):
        list_of_tfrecord_files.append(outpath+str(x)+'.tfrecord')

    dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)
    dataset.shuffle(10000, 42)
    # Save dataset to .tfrecord file
    filename = outpath+'.tfrecord.merge'
    
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)
    os.system('rm '+outpath+'*tfrecord')
if __name__=='__main__':
    path=sys.argv[1]
    num=10
    if len(sys.argv)==3:
        num=int(sys.argv[2])

    merge_record(path+'_train',num)
    merge_record(path+'_valid', num)


