import numpy as np
from Index_Builder import build_index,load_data,tokenize
import math
import copy


def language_model(vec_doc, invert_index):  # vec_doc没有删除停止词
    model_doc = copy.deepcopy(vec_doc)
    N = len(vec_doc)
    #  k:doc id
    for k in model_doc.keys():
        #  kk term name
        for kk in model_doc[k].keys():
            df = len(invert_index[kk])
            u_t = df/N
            p_t = 1/3 + 2/3 * u_t
            model_doc[k][kk] = math.log(p_t/(1-p_t), 10)+math.log((1-u_t)/u_t, 10)
    return model_doc


def cal_RSV(language_model, i, q):  # 语言模型下一个文档d(id为i)相对于查询q(q为terms字典)的得分
    RSV = 0
    for k in q.keys():
        if k in language_model[i]:
            RSV += language_model[i][k]
    return RSV



vector, query = load_data()
res_stem = tokenize(vector, 0)
invert_index, vec_doc = build_index(res_stem)
models_doc = language_model(vec_doc, invert_index)
for i in range(len(models_doc)):
    print(vec_doc[i+1])
    print(models_doc[i+1])

res_stem = tokenize(query, 0)
_, vec_query = build_index(res_stem)

print(cal_RSV(models_doc, 1, vec_query[1]))
