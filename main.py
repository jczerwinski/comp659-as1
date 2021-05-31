import nltk

# https://www.nltk.org/book/ch02.html
nltk.download('gutenberg')

words = nltk.corpus.gutenberg.words('austen-sense.txt')

# Conditional Frequency Distribution

# http://www.nltk.org/api/nltk.html?highlight=freqdist
cfd = nltk.probability.ConditionalFreqDist()

for word in words:
  outcome = len(word)
  condition = word[0].lower()
  cfd[condition][outcome] += 1

# What is the probability that a word that starts with "f" is a four-letter word?

print(cfd['f'].freq(4))

# Answer: 0.24937314793708684

# Stemming
# http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.wordnet
nltk.download('wordnet')

stemmer = nltk.stem.SnowballStemmer('english')

for word in words[60:80]:
  print(stemmer.stem(word) + ' is the stem of ' + word)

# Output:
# 
# general is the stem of general
# good is the stem of good
# opinion is the stem of opinion
# of is the stem of of
# their is the stem of their
# surround is the stem of surrounding
# acquaint is the stem of acquaintance
# . is the stem of .
# the is the stem of The
# late is the stem of late
# owner is the stem of owner
# of is the stem of of
# this is the stem of this
# estat is the stem of estate
# was is the stem of was
# a is the stem of a
# singl is the stem of single
# man is the stem of man
# , is the stem of ,
# who is the stem of who

# Tokenize
# http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize
nltk.download('punkt')
raw = nltk.corpus.gutenberg.raw('austen-sense.txt')

tokens = nltk.tokenize.word_tokenize(raw)

print(tokens[0:20])

# Output:
# ['[', 'Sense', 'and', 'Sensibility', 'by', 'Jane', 'Austen', '1811', ']', 'CHAPTER', '1', 'The', 'family', 'of', 'Dashwood', 'had', 'long', 'been', 'settled', 'in']

# Language Model

# POS Tagging
# http://www.nltk.org/api/nltk.tag.html#module-nltk.tag
 
nltk.download('averaged_perceptron_tagger')

print(nltk.tag.pos_tag(tokens[0:20]))

# Output:
# 
# [('[', 'JJ'), ('Sense', 'NNP'), ('and', 'CC'), ('Sensibility', 'NNP'), ('by', 'IN'), ('Jane', 'NNP'), ('Austen', 'NNP'), ('1811', 'CD'), (']', 'NNP'), ('CHAPTER', 'NNP'), ('1', 'CD'), ('The', 'DT'), ('family', 'NN'), ('of', 'IN'), ('Dashwood', 'NNP'), ('had', 'VBD'), ('long', 'RB'), ('been', 'VBN'), ('settled', 'VBN'), ('in', 'IN')]
