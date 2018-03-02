import pandas as pd
import string
import re


################################################################################################
## 前処理 ##

################################################################################################
## データ読み込み ##
datapath = './data/train.csv'
df_train = pd.read_csv(datapath, sep=',')
df_train.head()
#len(df_train)



# テキストを単語に分割
def data2words(csvdata):
    texts = []
    for text in csvdata["comment_text"].__iter__():
        text = re.sub(':', ' ', text)  # 改行を削除
        text = re.sub('_', ' ', text)  # 改行を削除
        text = re.sub('#', ' ', text)  # 改行を削除
        text = re.sub('\n', ' ', text)  # 改行を削除
        text = re.sub('\r', '', text)  # \rを削除
        text = re.sub('\xa0', ' ', text)  # \rを削除
        text = re.sub(r'[0-9]', '', text)  # 数字を削除
        text = re.sub('　', ' ', text)  # \rを削除

        text = text.lower()  # 小文字に
        text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation文字を削除

        texts.append(re.findall(r"[\w]+", text))  # 単語に分割
        #texts.remove('')

    return words_list


# テキストを単語に分割
def text_cleaning(csvdata):
    texts = []
    for text in csvdata["comment_text"].__iter__():
        text = re.sub(':', ' ', text)  # 改行を削除
        text = re.sub('_', ' ', text)  # 改行を削除
        text = re.sub('#', ' ', text)  # 改行を削除
        text = re.sub('\n', ' ', text)  # 改行を削除
        text = re.sub('\r', '', text)  # \rを削除
        text = re.sub('\xa0', ' ', text)  # \rを削除
        text = re.sub(r'[0-9]', '', text)  # 数字を削除
        text = re.sub('　', ' ', text)  # \rを削除

        text = text.lower()  # 小文字に
        text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation文字を削除

        #texts.append(re.findall(r"[\w]+", text))  # 単語に分割
        texts.append(text)

    return texts


#txts_train = data2words(df_train)
#txts_train[:100]

txts_train = text_cleaning(df_train)

################################################################################################
################################################################################################
## データ分割 ##

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(txts_train,
                                                    df_train['toxic'],
                                                    stratify=df_train['toxic'],
                                                    test_size=0.2,
                                                    random_state=1)

#len(X_train)
#len(X_test)

import gc
del df_train
del txts_train
gc.collect()


from gensim import corpora
from gensim.models import TfidfModel, LsiModel


# 訓練データのコーパス化とID化
corpus_train = list(map(lambda x:x.split(), X_train))  # 文章を単語で分割し，単語列(コーパス)に
corpus2id_dict_train = corpora.Dictionary(corpus_train) # 単語 -> id変換の辞書作成
#corpus2id_dict_train.token2id  #{単語：ID, ...}の辞書
id_corpus_train = [corpus2id_dict_train.doc2bow(sentence) for sentence in corpus_train]
#id_corpus_train  #[[(ID: 回数), ...], [(ID: 回数), ...], ...]
corpora.MmCorpus.serialize('./corpus/train.mm', id_corpus_train)  # corpusを保存


# テストデータのコーパス化とID化
corpus_test = list(map(lambda x:x.split(), X_test))  # 文章を単語で分割し，単語列(コーパス)に
corpus2id_dict_test = corpora.Dictionary(corpus_test) # 単語 -> id変換の辞書作成
#corpus2id_dict_train.token2id  #{単語：ID, ...}の辞書
#len(corpus2id_dict_train.token2id)  # 194936個の単語
id_corpus_test = [corpus2id_dict_test.doc2bow(sentence) for sentence in corpus_test]
#id_corpus_train  #[[(ID: 回数), ...], [(ID: 回数), ...], ...]
corpora.MmCorpus.serialize('./corpus/test.mm', id_corpus_test)  # corpusを保存


# 訓練データを用いたtfidfモデル
tfidfmodel_train = TfidfModel(id_corpus_train)
tfidf_train = tfidfmodel_train[id_corpus_train]

X_train[0]  # 各センテンスにおける[(単語ID, 値), ...]
tfidf_train[0]

# 訓練データで作ったtfidfモデルをテストデータに適用
tfidf_test = tfidfmodel_train[id_corpus_test]
X_test[0]
tfidf_test[0]


# LSIを用いて、194936次元から300次元まで圧縮
lsi_model = LsiModel(tfidf_train, id2word=corpus2id_dict_train, num_topics=300)
lsi_model.save('./corpus/lsi300.model')
lsi_corpus = lsi_model[tfidf_train]
len(lsi_corpus)
len(lsi_corpus[1])

l =[[(1,0),(2,0),(3,0)], [(1,0),(2,0),(3,0)], [(1,0),(2,0),(3,0)], [(1,0),(2,0),(3,0)]]
npl = [np.array(li).reshape(-1,2)[:,1] for li in l]
npl
np.array(npl).shape



X_train = [np.array(lsi_i).reshape(-1,2)[:,1] for lsi_i in lsi_corpus]
np.array(X_train).shape


X_train.shape
X_train
len(X_train)
len(X_train[0])
X_train[0]

import numpy as np
np.array(lsi_corpus).shape














##########################################################################################################
## sklearnのtfidfはメモリを大量に消費してしまう．．．

#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf_vectorizer = TfidfVectorizer(min_df=1, norm='l2')
#X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
#X_test = tfidf_vectorizer.transform(X_test).toarray()
#X_train
##########################################################################################################




################################################################################################
################################################################################################
## モデル学習 ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

clf = LogisticRegression(penalty='l2')
parameters = {'C': [10**(-i) for i in range(0,4)]}
scorer = make_scorer(f1_score)
grid_obj = GridSearchCV(clf, parameters, scorer, n_jobs=3)
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)
print("Fscore: %f" %(f1_score(y_test, predictions)))
print("Best Fscore: %f" %(f1_score(y_test, best_predictions)))


# モデルを保存する
filename = './model/logisticmodel_y1.sav'
pickle.dump(model, open(filename, 'wb'))

# 保存したモデルをロードする
#loaded_model = pickle.load(open(filename, 'rb'))
