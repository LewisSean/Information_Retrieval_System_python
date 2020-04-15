import numpy as np
from Index_Builder import build_index,load_data, tokenize
import math
# 一次只能返回一条查询向量与一个文档的匹配度
# vec_query是一个字典


def calculate_relevence(index_invert, vec_doc, vec_query, num_docs):
    num_terms = len(list(index_invert))
    terms_name = list(index_invert)
    terms = []

    q = np.zeros((1,num_terms))
    d = np.zeros((num_terms,1))
    for i in np.arange(num_terms):
        term = [terms_name[i]]  # word
        if terms_name[i] in vec_query:
            term.append(vec_query[terms_name[i]])  # tf
            term.append(vec_query[terms_name[i]])  # wf
            term.append(len(index_invert[terms_name[i]]))  # df
            term.append(math.log(num_docs / term[3], 10))  # idf
            term.append(term[2]*term[4])  # qi
        else:
            term += [0, 0]  # tf wf
            term.append(len(index_invert[terms_name[i]]))  # df
            term.append(math.log(num_docs / term[3], 10))  # idf
            term.append(0)  # qi
        terms.append(term)
        q[0, i] = term[5]
        print(term)

    docs = []
    base = 0
    print(terms_name)
    for i in np.arange(num_terms):
        doc = [terms_name[i]]  # word
        if terms_name[i] in vec_doc:
            doc.append(vec_doc[terms_name[i]])  # tf
            doc.append(math.log(vec_doc[terms_name[i]], 10)+1)  # wf
        else:
            doc += [0, 0]  # tf wf
        base += doc[1]**2
        docs.append(doc)

    base = math.sqrt(base)
    for i in np.arange(num_terms):
        docs[i].append(docs[i][2]/base)  # di
        print(docs[i])
        d[i, 0] = docs[i][3]

    return (np.dot(q, d) / (np.linalg.norm(d, 2) * np.linalg.norm(q, 2)))[0,0]




vector, query = load_data()
res_stem = tokenize(vector)
index_invert, vec_doc = build_index(res_stem)
res_stem = tokenize(query)
_, vec_query = build_index(res_stem)
print(vec_query[1])
print(vec_doc[1])
calculate_relevence(index_invert, vec_doc[1], vec_query[1], len(vec_doc))

