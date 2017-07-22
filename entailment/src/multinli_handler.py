import numpy as np
import sys
sys.path.append('../../src')
import data_io as dio

class MultiNliHandler(object):

    def __init__(self):
        self.label_idxs = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        self.headers = {}
        self.headers_n2i = {'promptID': 7, 'label1': 10, 'sentence1_binary_parse': 1, 'label3': 12,
                            'sentence2_parse': 4, 'label5': 14, 'label4': 13, 'sentence2_binary_parse': 2,
                            'sentence1': 5, 'sentence2': 6, 'sentence1_parse': 3, 'genre': 9, 'gold_label': 0,
                            'label2': 11, 'pairID': 8}
        self.headers_i2n = {0: 'gold_label', 1: 'sentence1_binary_parse', 2: 'sentence2_binary_parse',
                            3: 'sentence1_parse', 4: 'sentence2_parse', 5: 'sentence1', 6: 'sentence2', 7: 'promptID',
                            8: 'pairID', 9: 'genre', 10: 'label1', 11: 'label2', 12: 'label3', 13: 'label4',
                            14: 'label5'}

    def to_ignore(self, multinli_line):
        """
        clean data before parse, ignore label='-'
        :param multimli_line: dict of header:value
        :return:
        """
        if multinli_line['gold_label'] not in self.label_idxs.keys():
            return True
        return False

    def parse_header(self, multi_nli_file):
        """
        used to construct the headers n2i/i2n but doesn't reallly needed anymore
        :param multi_nli_file:
        :return:
        """
        f = open(multi_nli_file, 'r')
        header = f.readline().split('\t')
        self.headers_n2i = {name: i for i, name in enumerate(header)}
        self.headers_i2n = {i: name for i, name in enumerate(header)}

    def get_label_vector(self, label):
        label_vector = np.zeros(3)
        label_vector[self.label_idxs[label]] = 1.0
        return label_vector

    def parse_train_data(self, multi_nli_file, words):
        """
        read multinli txt fil, output array of words indices that can be fed into the alg.
        :param multi_nli_file: path to the txt file (not JSON!!!)
        :return:
        """

        golds = []
        seq1 = []
        seq2 = []

        f = open(multi_nli_file, 'r')
        f.readline() #skip header

        for line in f.readlines():
            sline = line.split('\t')
            line_dict = {self.headers_i2n[i]: val for i, val in enumerate(sline)}
            if self.to_ignore(line_dict):
                print 'dropped line ' + line_dict
                continue

            gold_label = self.get_label_vector(line_dict['gold_label'])
            sentence1 = line_dict['sentence1']
            sentence2 = line_dict['sentence2']

            golds.append(gold_label)
            s1_seq, s2_seq = dio.getSeqs(sentence1, sentence2, words)
            seq1.append(s1_seq)
            seq2.append(s2_seq)

        x1,m1 = dio.prepare_data(seq1)
        x2,m2 = dio.prepare_data(seq2)
        label_vectors =  np.stack(golds)
        return x1, m1, x2, m2, label_vectors
            # print line_dict
            # break
