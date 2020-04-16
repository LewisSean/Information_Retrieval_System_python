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


def cal_corp_model(invert_index):
    model_corp = {}
    sum_words = 0
    for k in invert_index.keys():
        sum_word = 0
        for kk in invert_index[k]:
            sum_word += invert_index[k][kk]
        model_corp[k] = sum_word
        sum_words += sum_word
        
    for k in model_corp.keys():
        model_corp[k] /= sum_words

    print(sum_words)
    return model_corp


def cal_doc_model(vec_doc):
    model_doc = copy.deepcopy(vec_doc)
    for k in model_doc.keys():
        sum_words = 0
        for kk in model_doc[k]:
            sum_words += model_doc[k][kk]
        for kk in model_doc[k]:
            model_doc[k][kk] /= sum_words
    return model_doc


def cal_LM(model_crop, model_doc, i, q):
    P_d_p = 0
    for k in q:
        if k in model_doc[i]:
            P_d_p += 0.5 * model_crop[k] + 0.5 * model_doc[i][k]
    P_d_p /= len(model_doc)  # 每个文件等概率出现
    return P_d_p

vector, query = load_data()
res_stem = tokenize(vector, 0)
invert_index, vec_doc = build_index(res_stem)
models_doc = language_model(vec_doc, invert_index)
for i in range(len(models_doc)):
    print(vec_doc[i+1])
    print(models_doc[i+1])

with open('language_model_1_RSV.txt','w') as f:
    for i in range(len(models_doc)):
        for k in models_doc[i+1]:
            models_doc[i + 1][k] = round(models_doc[i + 1][k], 6)
        f.writelines('D{doc_id:0>2d}'.format(doc_id=i+1)+str(models_doc[i+1])+'\n')
res_stem = tokenize(query, 0)
_, vec_query = build_index(res_stem)

print(cal_RSV(models_doc, 1, vec_query[1]))

model_corp = cal_corp_model(invert_index)
print(invert_index)
print(model_corp)

models_doc_2 = cal_doc_model(vec_doc)
print(vec_doc)
print(models_doc_2)

with open('language_model_Corpus.txt','w') as f:
    for (k,v) in model_corp.items():
        v = round(v,6)
        f.writelines('{k:<{len}} \t{v}'.format(k=k,v=v,len=10)+'\n')

with open('language_model_Docs.txt','w') as f:
    for (i,doc) in models_doc_2.items():
        f.writelines("==========Doc:{doc_id:0>2d}==========\n".format(doc_id=i))
        for (k,v) in doc.items():
            v = round(v,6)
            f.writelines('{k:<{len}} \t{v}'.format(k=k,v=v,len=10)+'\n')