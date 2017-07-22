import sys
sys.path.append('../../src')
import data_io as dio
import multinli_handler as mnh

multi_nli_file = '../data/multinli_0.9/multinli_0.9_train.txt'
wordfile = '../../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]

# load word vectors
(words, We) = dio.getWordmap(wordfile)

mlh = mnh.MultiNliHandler()
mlh.parse_train_data(multi_nli_file, words)
