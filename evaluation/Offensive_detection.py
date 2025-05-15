# coding=utf-8
import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
import jsonlines

tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')
model.eval()

# f = open('test_pangu.txt','r',encoding='utf-8').readlines()
# f = open('/home/omnisky/data2/data/wxx/GLM-4/data/result_offensive_GLM-4-sft-moral-offen.txt','r')
f = open('/data2/data/wxx/LLaMA-Factory/data/ccstories/result_qwen3_0.6b-ft-4000.txt','r')
sum = 0
# texts = ['1']
n=0
for text in f:
    n=n+1
    # post = text.find('标题：')
    # if post != -1:
    #     continue
    text = text[:300]
    # texts[0] = text
    model_input = tokenizer(text,return_tensors="pt",padding=True)
    model_output = model(**model_input, return_dict=False)
    prediction = torch.argmax(model_output[0].cpu(), dim=-1)
    prediction = [p.item() for p in prediction]
    sum += prediction[0]
    print(prediction)
print(n)
print(sum)


