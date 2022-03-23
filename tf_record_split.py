import tensorflow as tf
def split_record(path,shards=80):
    raw_dataset = tf.data.TFRecordDataset(path)

    

    for i in range(shards):
        if i ==shards-1:
            writer = tf.data.experimental.TFRecordWriter(
                f"valid.tfrecord")
        else:
            writer = tf.data.experimental.TFRecordWriter(
            f"output_file-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))
if __name__=="__main__":
    split_record('0.02_15_melspec_non_overlap_12800_train.tfrecord')
