from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

with open("tests/taylorswift.txt", "r") as file:
    text = file.read()

# basic_tokenizer = BasicTokenizer()
# basic_tokenizer.fit(text, vocab_size=2000, verbose=False)  #training the vocab

# #testing the trained tokenizer

# valtext = "Many common characters, including numerals123456789, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
valtext = "hello123!!!?     (안녕하세요!) 😉"
print(valtext)

# encoded = basic_tokenizer.encode(valtext)
# print("length of encoded ids with basic tokenizer:", len(encoded), "encoded:",encoded)
# decoded = basic_tokenizer.decode(encoded)
# # print(encoded)
# # print(decoded)
# print(valtext == decoded)

# regex_tokenizer = RegexTokenizer()
# regex_tokenizer.fit(text, vocab_size=5000, verbose=False)  #training the vocab
# regex_tokenizer.save("regex")

# regex_tokenizer_2 = RegexTokenizer()
# regex_tokenizer_2.load("regex.model")
# print(regex_tokenizer_2.pattern)
# print(regex_tokenizer_2.special_tokens)
# print(regex_tokenizer_2.merges)
# print(regex_tokenizer_2.vocab)

# encoded = regex_tokenizer.encode(valtext)
# print("length of encoded ids with regex tokenizer:", len(encoded), "encoded:",encoded)
# decoded = regex_tokenizer.decode(encoded)
# print(f"decoded: {decoded}")
# print(valtext == decoded)

gpt4_tokenizer = GPT4Tokenizer()
gpt4_tokenizer.save_vocab("models/gpt4.vocab")
# gpt4tok = GPT4Tokenizer()
# gpt4tok.load("models/gpt4.model")
# print(gpt4tok.pattern)
# print(gpt4tok.special_tokens)
# print(len(gpt4tok.merges))
# print(len(gpt4tok.vocab))

# encoded = gpt4tok.encode(valtext)
# print("length of encoded ids with gpt4 tokenizer:", len(encoded), "encoded:",encoded)
# decoded = gpt4tok.decode(encoded)
# print(f"decoded: {decoded}")
# print(valtext == decoded)