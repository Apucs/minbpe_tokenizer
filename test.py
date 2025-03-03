from minbpe.basic import BasicTokenizer

with open("tests/taylorswift.txt", "r") as file:
    text = file.read()

tokenizer = BasicTokenizer()
tokenizer.fit(text, vocab_size=2000, verbose=True)  #training the vocab

#testing the trained tokenizer

valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."

encoded = tokenizer.encoder(valtext)
decoded = tokenizer.decoder(encoded)
print(valtext)
# print(encoded)
# print(decoded)
print(valtext == decoded)