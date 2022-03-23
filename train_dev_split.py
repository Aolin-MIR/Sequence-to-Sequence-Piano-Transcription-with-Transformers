import tensorflow as tf


# Create dataset from multiple .tfrecord files
def merge_record(outpath,num=80):
    list_of_tfrecord_files=[]
    for x in range(num-1):
        list_of_tfrecord_files.append('output_file-part-'+str(x)+'.tfrecord')
    
    dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)
    dataset.shuffle(10000,42)
    # Save dataset to .tfrecord file
    filename = outpath
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)
if __name__ =='__main__':
    # merge_record()
    pass