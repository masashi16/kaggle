import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Embedding, Input, Dense, LSTM, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, one_hot
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping


# 分析の流れ
# 全文章を使って，word2vecを学習
# 単語列(seq)を入力として，biLSTMで学習
#   - 入力長を選定：　
#   - 全文章を単語列に分ける → それぞれをone-hot表現にする
#   - 各documentを単語列


trainpath = './data/train.csv'
testpath = './data/test.csv'

df_train = pd.read_csv(trainpath)
df_test = pd.read_csv(testpath)

df_train.iloc[0]


all_corpus = df_train['comment_text'].append(df_test['comment_text'])
all_tokens = [gensim.utils.simple_preprocess(txt) for txt in all_corpus]
tokens_train = [gensim.utils.simple_preprocess(txt) for txt in df_train['comment_text']]
tokens_test = [gensim.utils.simple_preprocess(txt) for txt in df_test['comment_text']]

# word2vec training
#w2v_model = Word2Vec(all_tokens, size=100, min_count=1, negative=20)
#w2v_model.save('./emb/w2v_100.model')

# word2vec load
w2v_model = Word2Vec.load('./emb/w2v_100.model')


#maxlen = 0
#for tokens in all_tokens:
#    if len(tokens) > maxlen:
#        maxlen = len(tokens)
#maxlen

length = [len(tokens) for tokens in all_tokens]
pd.Series(length).hist()
plt.show()

maxlen = int(stats.scoreatpercentile(length, 90))
#maxlen



tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)
all_seq = tokenizer.texts_to_sequences(all_corpus)  # 単語ID列に
seq_train = tokenizer.texts_to_sequences(df_train['comment_text'])
seq_test = tokenizer.texts_to_sequences(df_test['comment_text'])

seq_pad_train = sequence.pad_sequences(seq_train, maxlen=maxlen, padding='pre', truncating='pre')
seq_pad_test = sequence.pad_sequences(seq_test, maxlen=maxlen, padding='pre', truncating='pre')



word_index = tokenizer.word_index  # {単語: ID, ...}の変換辞書

embedding_matrix = np.zeros((len(word_index)+1, 100))  # paddingは0で埋めるので，その0番目の要素分のプラス１？
for word, i in word_index.items():
    if word in w2v_model.wv.vocab:
        embedding_matrix[i] = w2v_model.wv[word]  # word2vecモデルに含まれていない単語はゼロベクトルとする


def tokenIDs2onehot(wordID_list, word_index):
    vecs = []
    for id in wordID_list:
        vec = np.zeros(len(word_index))
        vec[id] = 1
        vecs.append(vec)
    return vecs

X_train = [tokenIDs2onehot(seq, word_index) for seq in seq_pad_train]


#one_hot('slkgjsdb', word_index)


embedding_layer = Embedding(len(word_index)+1, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)

sequence_input = Input(shape=(maxlen*100,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(128, activation='relu', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', dropout=0.2, recurrent_dropout=0.2, use_bias=True, bias_initializer='zeros', unit_forget_bias=True, return_sequences=True))(embedded_sequences)
x = Dropout(rate=0.3)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(input=sequence_input, output=preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()


y1_train = np.array(df_train.iloc[:,2])
y1_train

#to_categorical(y1_train)

#X_train[3]

model.fit(X_train, y1_train, batch_size=1024, epochs=100, verbose=1, validation_split=0.8, callbacks=[EarlyStopping()])
