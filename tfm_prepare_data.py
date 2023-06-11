import os
import errno
import sentencepiece as spm
import re
from tfm_hparams import Hparams
import logging

def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    train1 = "data/sample50w_paracrawl.wmt21.zh"
    train2 = "data/sample50w_paracrawl.wmt21.en"

    for f in (train1, train2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    # train
    #_prepro = lambda x:  [line.strip() for line in open(x, 'r').read().split("\n") \
    #                  if not line.startswith("<")]
    _prepro = lambda x:  [ line.strip() for line in open(x, 'r').read().split("\n") ]
    prepro_train1, prepro_train2 = _prepro(train1), _prepro(train2)
    assert len(prepro_train1)==len(prepro_train2), "Check if train source and target files match."


    logging.info("Let's see how preprocessed data look like")
    logging.info("prepro_train1:", prepro_train1[0])
    logging.info("prepro_train2:", prepro_train2[0])

    logging.info("# write preprocessed files to disk")
    os.makedirs("data/prepro", exist_ok=True)
    def _write(sents, fname):
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "data/prepro/train.zh")
    _write(prepro_train2, "data/prepro/train.en")
    _write(prepro_train1+prepro_train2, "data/prepro/train")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("data/segmented", exist_ok=True)
    train = '--input=data/prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=data/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("data/segmented/bpe.model")

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "data/segmented/train.zh.bpe")
    _segment_and_write(prepro_train2, "data/segmented/train.en.bpe")

    logging.info("Let's see how segmented data look like")
    print("train1:", open("data/segmented/train.zh.bpe",'r').readline())
    print("train2:", open("data/segmented/train.en.bpe", 'r').readline())


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
