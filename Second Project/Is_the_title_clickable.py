import codecs
import re
import spacy

# import keras
import numpy as np
import json

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

file = codecs.open("../all_data 3.json", "r", "utf-8")

json_str = file.read()
json_data = json.loads(json_str)

data_features = np.array([len(json_data)])
data_label = np.array([len(json_data)])

# re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
#
#
# def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# tokenize = keras.preprocessing.text.Tokenizer(num_words=None,
#                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ',
#                                               char_level=False, oov_token=None, document_count=0)
# the clicks that has clicked more than 1 from a unique user
count = 0
for id in json_data:
    if str(id['event_type']) == "CLICK":
        label = 1;
    else:
        label = 0;

    title = str(id["ad_title"])

    # tokenized = keras.preprocessing.text.text_to_word_sequence(title,
    #                                                           filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
    #                                                           lower=True, split=' ')
    data_features[count] = np.append(data_features[count], title)
    data_label = np.append(data_label, label)
    count += 1
    # re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    # def tokenize(title):
    #     return re_tok.sub(r' \1 ', title).split()

X_train, X_test, y_train, y_test = train_test_split(data_features,
                                                    data_label,
                                                    test_size=0.2)
nlp = spacy.load('en_core_web_sm')
vect = CountVectorizer(tokenizer=nlp)
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)
p = tf_train[y_train == 1].sum(0) + 1
q = tf_train[y_train == 0].sum(0) + 1
r = np.log((p / p.sum()) / (q / q.sum()))
b = np.log(len(p) / len(q))

pre_preds = tf_test @ r.T + b
preds = pre_preds.T > 0
accuracy = (preds == y_test).mean()
# if not os.path.exists("Dataset"):
#     os.makedirs("Dataset")
# np.save("Dataset//features.npy", data_features)
# np.save("Dataset//labels.npy", data_label)