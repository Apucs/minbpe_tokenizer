from minbpe import BasicTokenizer, RegexTokenizer

with open("tests/taylorswift.txt", "r") as file:
    text = file.read()

basic_tokenizer = BasicTokenizer()
basic_tokenizer.fit(text, vocab_size=5000, verbose=False)  #training the vocab

#testing the trained tokenizer

# valtext = "Many common characters, including numerals123456789, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
valtext = "hello123!!!? (안녕하세요!) 😉"
print(valtext)

encoded = basic_tokenizer.encoder(valtext)
print("length of encoded ids with basic tokenizer:", len(encoded), "encoded:",encoded)
decoded = basic_tokenizer.decoder(encoded)
# print(encoded)
# print(decoded)
print(valtext == decoded)

regex_tokenizer = RegexTokenizer()
regex_tokenizer.fit(text, vocab_size=5000, verbose=False)  #training the vocab
encoded = regex_tokenizer.encoder(valtext)
print("length of encoded ids with regex tokenizer:", len(encoded), "encoded:",encoded)
decoded = regex_tokenizer.decoder(encoded)
print(f"decoded: {decoded}")
print(valtext == decoded)