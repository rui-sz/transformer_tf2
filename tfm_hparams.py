import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    parser.add_argument('--epoch', default=6, type=int)
    parser.add_argument('--buffer_size', default=20000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--maxlen', default=60, type=int, help="maximum length of a source sequence")
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--train_src1', default='data/sample50w_paracrawl.wmt21.zh',
                             help="chinese training data")
    parser.add_argument('--train_src2', default='data/sample50w_paracrawl.wmt21.en',
                             help="english training data")

    parser.add_argument('--train1', default='data/segmented/train.zh.bpe',
                             help="chinese training segmented data")
    parser.add_argument('--train2', default='data/segmented/train.en.bpe',
                             help="english training segmented data")


    # model
    parser.add_argument('--num_layers', default=8, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_model', default=256, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=1024, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")

    #########################################################################################################

    # train
    ## files
    parser.add_argument('--eval1', default='iwslt2016/segmented/eval.de.bpe',
                             help="german evaluation segmented data")
    parser.add_argument('--eval2', default='iwslt2016/segmented/eval.en.bpe',
                             help="english evaluation segmented data")
    parser.add_argument('--eval3', default='iwslt2016/prepro/eval.en',
                             help="english evaluation unsegmented data")

    ## vocabulary
    parser.add_argument('--vocab', default='iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")