# -*- coding:utf-8 -*-
import scipy
from bleurt import score

checkpoint = "./BLEURT-20"
scorer = score.BleurtScorer(checkpoint)

scores=0
fp=open('../test_content.txt','r',encoding="utf-8")
# fq=open('/home/omnisky/data2/data/wxx/GLM-4/data/result_GLM-4-sft-lot.txt','r',encoding='utf-8')
fq=open('/data2/data/wxx/LLaMA-Factory/data/ccstories/result_qwen3_0.6b-ft-4000.txt')
line1 = fp.readlines()
line2 = fq.readlines()
for i in range(1000):
    candidates = [line2[i]]
    references = [line1[i]]
    print(i)
    scores += scorer.score(references=references, candidates=candidates)[0]
print("======")
print(scores)







