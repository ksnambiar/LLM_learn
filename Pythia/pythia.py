from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-m-deduped",
  revision="step113000",
  cache_dir="./pythia-70m-deduped/step113000",
)

def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = model.generate(**inputs)
    result = tokenizer.decode(tokens[0])
    print("\nAnswer start ------\n")
    print(result)
    print("\nAnswer end ------\n")


# generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

# res = generate_text("My name is Sidharth. What username would you suggest for me ? i need a classy one.")
# print(res[0]["generated_text"])

# res = generate_text("Can you give two points in concise ?")
# print(res[0]["generated_text"])
generate_text("My name is Sidharth.")

generate_text("Can you give two points in concise ?")