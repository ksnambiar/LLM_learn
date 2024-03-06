# import torch
# from transformers import pipeline

'''
There are three models

1. dolly-v2-3b : 2.8B parameters
2. dolly-v2-7b : 6.9B parameters
3. dolly-v2-12b : Most useful model
'''

# generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload", offload_state_dict = True)



import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, offload_folder="offload")

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

# print("Replied with - {}\n".format() )
res = generate_text("My name is Sidharth. What username would you suggest for me ? i need a classy one.")
print(res[0]["generated_text"])

res = generate_text("Can you give two points in concise ?")
print(res[0]["generated_text"])