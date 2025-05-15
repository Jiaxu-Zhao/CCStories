#-*- coding:utf-8 -*-

from comet import download_model, load_from_checkpoint
# model_path = download_model("eamt22-cometinho-da",'BLEURT-20/COMET-22')
# model_path = 'unbabel_comet/wmt20-comet-da'
model_path = 'XCOMET-XL/checkpoints/model.ckpt'
# model_path = 'BLEURT-20/COMET-22/eamt22-cometinho-da/checkpoints/model.ckpt'
model_output = 0
fp=open('../test_content.txt','r',encoding="utf-8")
# fq=open('/home/omnisky/data2/data/wxx/GLM-4/data/result_GLM-4-sft-cc.txt','r',encoding='utf-8')
fq=open('/data2/data/wxx/LLaMA-Factory/data/ccstories/result_qwen3_0.6b-ft-4000.txt','r',encoding='utf-8')
line1 = fp.readlines()
line2 = fq.readlines()
model = load_from_checkpoint(model_path)
for i in range(100):
    src = [line1[i]]
    mt = [line2[i][20:]]
    ref = [line1[i]]
    data = [
        {
            "src": f"{src}",
            "mt": f"{mt}",
            "ref": f"{ref}"
        }
    ]
    print(i)
    model_output += model.predict(data, batch_size=8, gpus=1).scores[0]
print("======")
print(model_output)

# data = [
#     {
#         "src": "Dem Feuer konnte Einhalt geboten werden",
#         "mt": "The fire could be stopped",
#         "ref": "They were able to control"
#     },
#     #-0.794
#     # {
#     #     "src": "Schulen und Kindergärten wurden eröffnet.",
#     #     "mt": "Schools and kindergartens were open",
#     #     "ref": "Schools and kindergartens opened"
#     # },
#     #1.207

# ]
# model_output = model.predict(data, batch_size=8, gpus=1).scores
# print(model_output)