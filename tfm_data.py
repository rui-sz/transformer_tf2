
import tensorflow_datasets as tfds
import tensorflow as tf

def load_data():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
    return examples, metadata


def load_local_data( fpath1, fpath2, maxlen1, maxlen2 ):
    sents1, sents2 = [], []
    total_len = 0
    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
            total_len += 1

    examples = {}
    examples["train"] = []
    examples["validation"] = []

    train_len = tf.cast(3*total_len/4,tf.int32)
    print(train_len)
    for sent1, sent2 in zip(sents1[:train_len],sents2[:train_len]):
        examples["train"].append((sent1,sent2))

    for sent1, sent2 in zip(sents1[train_len+1:],sents2[train_len+1:]):
        examples["validation"].append((sent1,sent2))

    return examples
