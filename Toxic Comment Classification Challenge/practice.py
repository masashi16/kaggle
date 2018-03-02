import pandas as pd
import string


from gensim.models import Word2Vec
from gensim.models import word2vec
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM

from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train')
labels = twenty_train.target
texts = []
for doc in twenty_train.data:
    i = doc.find('\n\n')  # skip header
    texts.append(doc[i:])

texts[:10]

w2v = word2vec.Word2Vec(min_count=1)
w2v.build_vocab([doc.split(' ') for doc in texts])
w2v.train(texts, total_examples=w2v.corpus_count, epochs=w2v.iter)

embedding_layer = w2v.wv.get_keras_embedding(train_embeddings=True)

MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_train = to_categorical(labels)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
preds = Dense(y_train.shape[1], activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

model.fit(x_train, y_train, epochs=10)



import pandas as import pd

datapath = './data/train.csv'
df_train = pd.read_csv(datapath, sep=',')





string.punctuation

t = "Daww! He matches this background colour I'm seemingly stuck with. Thanks  talk 21:51, January 11, 2016 (UTC)"

t = t.translate(str.maketrans('', '', string.punctuation))
t
#t = t.lstrip('  ')

l = re.findall(r"[\w]+", t)
l




################################################################################################
## word2vec ##
from gensim.models import word2vec

model = word2vec.Word2Vec(txts_train, size=300, min_count=1)
model.save('./emb/toxic_comment_w2v.model')
model.wv.save_word2vec_format('./emb/toxic_comment_w2v.emb')



################################################################################################
## LSTM ##
# 6つのtoxicな要素を，それぞれ予測(判定)する6つのモデルを作成
# 各文章に対して，各要素が1となる確率を求める
# 注意として，テストデータには正解が含まれていない



from gensim.models import Word2Vec
from gensim.models import word2vec
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM


#model = Word2Vec.load('./emb/toxic_comment_w2v.model')





"""
datapath = './data/train.csv'
df_train = pd.read_csv(datapath, sep=',')
txts_train = [i for i in df_train.iloc[:,1].__iter__()]




w2v = word2vec.Word2Vec(size=200, min_count=1, negative=20)
w2v.build_vocab([doc.split(' ') for doc in txts_train])  # Build vocabulary from a sequence of sentences
w2v.train(txts_train, total_examples=w2v.corpus_count, epochs=w2v.iter)
w2v.save('./emb/toxic_comment_w2v.model')
w2v.wv.save_word2vec_format('./emb/toxic_comment_w2v.emb')
# 引数 train_embeddings: Trueだとモデルの学習中にEmbedding層を学習する，Falseだと学習しない
"""


######################################################################################
# w2vで，それを特徴量としてロジスッティック回帰
w2vmodel = Word2Vec.load('./emb/toxic_comment_w2v.model')

txts = ['explanation']
txts[0]
w2vmodel[txts[0]]

txts_train[0][0]
w2vmodel[txts_train[0][0]]


txts_train[:10]
txts_train[0][0]

import operator
for txt in txts_train:
    txt_w2v = [w2vmodel[txt_i] for txt_i in txt]
    #for i in range(len(txt)):
    #    if i+1 > len(txt): break
    #    txt_w2v +=  w2vmodel[txt[i+1]]












######################################################################################











embedding_layer = w2v.wv.get_keras_embedding(train_embeddings=True)


#txts_train[:10]
#txts_train = data2words(df_train)

#max_num_words = 0
#for txt in txts_train:
#    if len(txt) > max_num_words:
#        max_num_words = len(txt)

#max_num_words
#print(txts_train[:10])

#txts_train = [i for i in df_train.iloc[:,1].__iter__()]
#txts_train[:10]






MAX_SEQUENCE_LENGTH = 2000



# Tokenizer を用いて、単語列を単語ID列に変換し、パディングする
tokenizer_train = Tokenizer()
tokenizer_train.fit_on_texts(txts_train)  # 単語にIDをふる
sequences_train = tokenizer_train.texts_to_sequences(txts_train)  # 単語列をID列にする
#sequences_train[:10]

x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
x_train.shape

y1_train = to_categorical([i for i in df_train.iloc[:,2].__iter__()])
y2_train = to_categorical([i for i in df_train.iloc[:,3].__iter__()])
y3_train = to_categorical([i for i in df_train.iloc[:,4].__iter__()])
y4_train = to_categorical([i for i in df_train.iloc[:,5].__iter__()])
y5_train = to_categorical([i for i in df_train.iloc[:,6].__iter__()])
y6_train = to_categorical([i for i in df_train.iloc[:,7].__iter__()])
#print(y1_train[:10])



testdata_path = './data/test.csv'
df_test = pd.read_csv(testdata_path, sep=',')
df_test.head()

#txts_test = data2words(df_test)
txts_test = [i for i in df_test.iloc[:,1].__iter__()]

tokenizer_test = Tokenizer()
tokenizer_test.fit_on_texts(txts_test)
sequences_test = tokenizer_test.texts_to_sequences(txts_test)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

df_test["id"]

#y1_train.shape[1]


## モデル構築 ##







## 学習 ##

#n_hidden = 128
#timesteps = 20
#n_in = max_num_words
#n_out = 1

#batch_size = 300
#epochs = 10

#MAX_SEQUENCE_LENGTH = 200

x_train.shape

"""
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH ,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
preds = Dense(y1_train.shape[1], activation='sigmoid')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
"""

def create_model():
    model = Sequential()
    #sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH ,), dtype='int32')
    #model.add(Input(shape=(MAX_SEQUENCE_LENGTH ,), dtype='int32'))
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y1_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

model1 = create_model()
model1.fit(x_train, y1_train, batch_size=300, epochs=3, validation_split=0.3)
model1.save_weights('./model/param2_y1.hdf5')


#score = model.evaluate(x_test, y1_train)
#print(score)

#prob = model.predict_proba(x_test)
#print(prob)


model2 = create_model()

model2.fit(x_train, y2_train, batch_size=300, epochs=3, validation_split=0.3)
model2.save_weights('./model/param2_y2.hdf5')

model3 = create_model()
model3.fit(x_train, y3_train, batch_size=300, epochs=3, validation_split=0.3)
model3.save_weights('./model/param2_y3.hdf5')

model4 = create_model()
model4.fit(x_train, y4_train, batch_size=300, epochs=3, validation_split=0.3)
model4.save_weights('./model/param2_y4.hdf5')

model5 = create_model()
model5.fit(x_train, y5_train, batch_size=300, epochs=3, validation_split=0.3)
model5.save_weights('./model/param2_y5.hdf5')

model6 = create_model()
model6.fit(x_train, y6_train, batch_size=300, epochs=3, validation_split=0.3)
model6.save_weights('./model/param2_y6.hdf5')



####################################################################################################
####################################################################################################

model1.load_weights('./model/param2_y1.hdf5', by_name=False)

predict1 = model1.predict(x_test)
predict1.sum(axis=1)
pre = model1.predict_proba(x_test)
pre.sum(axis=1)

y1_train
model1.predict_classes(x_test)

pr_train1 = model1.predict(x_train)
pr_train1

pre.sum(axis=1).shape
predict1.sum(axis=1)

model2.load_weights('./model/param2_y2.hdf5', by_name=False)
predict2 = model2.predict_proba(x_test)
model3.load_weights('./model/param2_y3.hdf5', by_name=False)
predict3 = model3.predict_proba(x_test)
model4.load_weights('./model/param2_y4.hdf5', by_name=False)
predict4 = model4.predict_proba(x_test)
model5.load_weights('./model/param2_y5.hdf5', by_name=False)
predict5 = model5.predict_proba(x_test)
model6.load_weights('./model/param2_y6.hdf5', by_name=False)
predict6 = model6.predict_proba(x_test)


import numpy as np
predict_ar = np.array([predict1[:,1], predict2[:,1], predict3[:,1], predict4[:,1], predict5[:,1], predict6[:,1]]).reshape(-1,6)
predict_ar.shape

y1_train
y1_train.sum(axis=1)
predict1.sum(axis=1).shape
predict_ar.shape
df = pd.DataFrame(predict_ar, columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"])

df.index = df_test["id"]
df.to_csv('./result/result.csv')

#バックエンドをインポート
#from keras.backend import tensorflow_backend as backend
#処理終了時に下記をコール
#backend.clear_session()
