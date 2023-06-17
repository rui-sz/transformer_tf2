
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

    x, y = zip(*examples["train"])
    examples["train"] = tf.data.Dataset.from_tensor_slices((list(x), list(y)))

    x, y = zip(*examples["validation"])
    examples["validation"] = tf.data.Dataset.from_tensor_slices((list(x), list(y)))

    return examples

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    print("head 10 of vocab: ", vocab[0:10])
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}

    return token2idx, idx2token


def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)


def tokenization( sentence, tokenizer_token2idx, tokenizer_idx2token, enc=True ):
    sent_str = sentence
    if enc==True:
        tokens_lang1 = sent_str.split()
        print(tokens_lang1)
        return [tokenizer_token2idx.get(t, tokenizer_token2idx["<unk>"]) for t in tokens_lang1]
    else:
        tokens_lang2 = sent_str
        return [tokenizer_idx2token.get(t, "<unk>") for t in tokens_lang2]
