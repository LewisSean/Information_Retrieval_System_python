import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def tokenize(vector, flag=1):  # flag = 1则去掉停止词
    res_stem = []
    stemmer = PorterStemmer()
    for str in vector:
        str = str.lower()
        str = re.sub(r'\d +', '', str)
        str = re.sub(r'[^\w\s]', ' ', str)
        tokens = word_tokenize(str)
        if flag == 1:
            stop_words = set(stopwords.words('english'))
            result = [i for i in tokens if not i in stop_words]
        else:
            result = tokens
        result_stem = []
        for word in result:
            result_stem.append(stemmer.stem(word))
        res_stem.append(result_stem)

    return res_stem


    '''
    print("===============================================================")
    for i in res_lemma:
        print(i)
    '''
def build_index(res_stem):
    index_invert = {}
    num_doc = len(res_stem)
    vec_doc = {}
    id = 1
    for item in res_stem:
        for i in item:
            if not id in vec_doc:
                vec_doc[id] = {}
            if not i in vec_doc[id]:
                vec_doc[id][i] = 0
            vec_doc[id][i] += 1

            if not i in index_invert:
                index_invert[i] = {}
            if not id in index_invert[i]:
                index_invert[i][id] = 0
            index_invert[i][id] += 1
        id += 1

    return index_invert, vec_doc
'''
    for (k, v) in index_invert.items():
        print(k, v)

    for (k, v) in vec_doc.items():
        print(k, v)
'''


def load_data():
    vector = []
    query = []
    with open('docId.txt', 'r') as f:
        for line in f.readlines():
            vector.append(line.strip())
    with open('query.txt', 'r') as f:
        for line in f.readlines():
            query.append(line.strip())

    return vector,query



