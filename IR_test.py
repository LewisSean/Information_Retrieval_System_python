from Index_Builder import build_index,load_data, tokenize
from Relevence_Ranking import calculate_relevence
from Language_Model import cal_RSV, language_model,cal_corp_model,cal_doc_model,cal_LM
import csv
vector, query = load_data()
print("\n\n================================Vector Model=========================================\n\n")
res_stem = tokenize(vector)
index_invert, vec_doc = build_index(res_stem)
res_stem = tokenize(query)
_, vec_query = build_index(res_stem)
print(vec_query[1])
print(vec_doc[1])

rank_VM = {}
for i in range(len(vec_doc)):
    score = calculate_relevence(index_invert, vec_doc[i+1], vec_query[1], len(vec_doc))
    if score in rank_VM:
        rank_VM[score].append(i+1)
    else:
        rank_VM[score] =[i+1]
list_rank = sorted(rank_VM)
str_rank = []
list_final = []
for scores in list_rank:
    for id in rank_VM[scores]:
        str_rank.append('D{doc_id:0>2d} socre: {score}'.format(doc_id=id, score=scores))
        list_final.append((id, vector[id-1], scores))

list_final.reverse()
str_rank.reverse()
for i in str_rank:
    print(i)

headers = ['doc_ID','doc_name','score']


with open('rank_VM.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(list_final)

print("\n\n================================Language Model(RSV)=========================================\n\n")

res_stem = tokenize(vector, 0)
invert_index, vec_doc = build_index(res_stem)
models_doc = language_model(vec_doc, invert_index)
for i in range(len(models_doc)):
    print(vec_doc[i+1])
    print(models_doc[i+1])

res_stem = tokenize(query, 0)
_, vec_query = build_index(res_stem)

print(cal_RSV(models_doc, 1, vec_query[1]))

rank_RSV = {}
for i in range(len(vec_doc)):
    score = cal_RSV(models_doc, i+1, vec_query[1])
    if score in rank_RSV:
        rank_RSV[score].append(i + 1)
    else:
        rank_RSV[score] =[i + 1]
list_rank = sorted(rank_RSV)
str_rank = []
list_final = []
for scores in list_rank:
    for id in rank_RSV[scores]:
        str_rank.append('D{doc_id:0>2d} socre: {score}'.format(doc_id=id, score=scores))
        list_final.append((id, vector[id-1], scores))

list_final.reverse()
str_rank.reverse()
for i in str_rank:
    print(i)

headers = ['doc_ID','doc_name','score']


with open('rank_RSV.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(list_final)

print("\n\n================================Language Model=========================================\n\n")

res_stem = tokenize(vector, 0)
invert_index, vec_doc = build_index(res_stem)
model_doc_2 = cal_doc_model(vec_doc)
model_corp = cal_corp_model(invert_index)
for i in range(len(model_doc_2)):
    print(vec_doc[i+1])
    print(model_doc_2[i+1])

res_stem = tokenize(query, 0)
_, vec_query = build_index(res_stem)

rank_LM = {}
for i in range(len(vec_doc)):
    score = cal_LM(model_corp, model_doc_2, i+1, vec_query[1])
    if score in rank_LM:
        rank_LM[score].append(i + 1)
    else:
        rank_LM[score] =[i + 1]
list_rank = sorted(rank_LM)
str_rank = []
list_final = []
for scores in list_rank:
    for id in rank_LM[scores]:
        str_rank.append('D{doc_id:0>2d} socre: {score}'.format(doc_id=id, score=scores))
        list_final.append((id, vector[id-1], scores))

list_final.reverse()
str_rank.reverse()
for i in str_rank:
    print(i)

headers = ['doc_ID','doc_name','score']


with open('rank_LM.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(list_final)
