#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow_datasets as tfds
import tensorflow as tf
import sentencepiece as spm

import time
import numpy as np
import matplotlib.pyplot as plt

from tfm_model import Encoder, Decoder, Transformer, CustomSchedule, create_padding_mask, create_look_ahead_mask
from tfm_hparams import Hparams
from tfm_data import load_data, load_local_data, load_vocab, tokenization


hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


# ==================================================================================================================
# step1, 加载原始数据，tokenization

# examples
#examples, metadata = load_data()
fpath1="data/sample50w_paracrawl.wmt21.zh"
fpath2="data/sample50w_paracrawl.wmt21.en"
maxlen1=60
maxlen2=60
examples = load_local_data(fpath1,fpath2,maxlen1,maxlen2)
train_examples, val_examples = examples['train'], examples['validation']

# tokens
vocab_fpath = "data/segmented/bpe.vocab"
tokenizer_token2idx, tokenizer_idx2token = load_vocab(vocab_fpath)

vocab_file = "data/segmented/bpe.model"
sp = spm.SentencePieceProcessor()
sp.load(vocab_file)

sample_string = '我在坡县等你来！'
tokenized_string = sp.encode_as_ids(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))
print ("decoded string is: ", sp.decode_ids(tokenized_string))


# ==================================================================================================================
# step，处理 train 和 val 数据集

def encode(lang1, lang2):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    lang1_str = lang1.numpy().decode("utf-8")
    lang2_str = lang2.numpy().decode("utf-8")

    # 不同的开始和结束标记
    lang1_lst = [hp.vocab_size] + sp.encode_as_ids(lang1_str) + [hp.vocab_size+1]
    lang2_lst = [hp.vocab_size+2] + sp.encode_as_ids(lang2_str) + [hp.vocab_size+3]

    return lang1_lst, lang2_lst

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


BUFFER_SIZE = hp.buffer_size
BATCH_SIZE = hp.batch_size
MAX_LENGTH = hp.maxlen   # 字符串最大长度

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

# tf_encode 的入参与 train_examples 中元素类型匹配
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# 将数据集缓存到内存中以加快读取速度。
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=53).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

for tmp in train_dataset.take(2):
    pt, en = tmp
    print(pt, en)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)


# ==================================================================================================================
# step，创建 Transformer 对象

num_layers = hp.num_layers      # transformer 的层数
d_model = hp.d_model       # emb 维数
dff = hp.d_ff           # ？
num_heads = hp.num_heads       # MHA 头的个数

input_vocab_size = hp.vocab_size + 4
target_vocab_size = hp.vocab_size + 4
dropout_rate = 0.1

print("after config hyperparameters, create optimizer")

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

print("after create optimizer, loss function and metrics")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    # 将填充的部分标记为0，非填充的部分标记为1，计算损失只关注非填充部分
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, 
                          target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')


print("===============before evaluate")

'''
# ==================================================================================================================
# step9, train

EPOCHS = 1

# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
  
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))
    
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

'''

# ==================================================================================================================
# step10, evaluate

print("evaluate")

def evaluate(inp_sentence):
    # 输入语句是葡萄牙语，增加开始和结束标记
    inp_sentence = [hp.vocab_size] + sp.encode_as_ids(inp_sentence) + [hp.vocab_size + 1]
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # 因为目标是英语，输入 transformer 的第一个词应该是英语的开始标记。
    decoder_input = [hp.vocab_size+2]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == hp.vocab_size+3:
            return tf.squeeze(output, axis=0), attention_weights

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)
  
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # 画出注意力权重
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result)-1.5, -0.5)
            
        ax.set_xticklabels(
            ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
                            if i < tokenizer_en.vocab_size], 
                            fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head+1))
  
    plt.tight_layout()
    plt.show()


# ==================================================================================================================
# step11, test

def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    res_arr = [ int(i) for i in result.numpy() if i<hp.vocab_size ]
    predicted_sentence = sp.decode_ids( res_arr )  

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)

translate("这真的是一个我们面临的问题")
print ("Real translation: this is a problem we have to solve.")

translate("我邻居的家庭听到了这个主意")
print ("Real translation: and my neighboring homes heard about this idea.")

translate("所以我刚刚给你们分享了一些很神奇的故事")
print ("Real translation: so i'll just share with you some stories very quickly of some magical things that have happened .")

#translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
translate("这是我完成写作的第一本书")
print ("Real translation: this is the first book i've ever done.")

