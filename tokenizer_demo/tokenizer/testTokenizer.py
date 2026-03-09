from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

output = tokenizer.encode("I love machine learning")

print(output.tokens)
print(output.ids)