import sys
import math
import argparse
import hashlib, csv, os, pickle, subprocess
from torch.utils import data
import pdb
import torch


class Data(data.Dataset):
    def __init__(self, train=None, index_emebeding=None):
        self.train = train
        self.embed = index_emebeding
        self.feature_sizes, self.datas = self.read_data()
        # pdb.set_trace()
        self.length = len(self.datas)

    
    def __getitem__(self, index):
        index = index % self.length
        label, ind, score = self.preprocess(self.datas[index])

        return label, ind, score
    
    def __len__(self):
        return self.length
    
    def load_category_index(self):
        with open(self.embed, 'r') as f:
            data = f.readlines()
            # pdb.set_trace()
            self.field_sizes = int(data[-1].split(":")[0])
            cate_dict = [0] * (self.field_sizes + 1)
            for l in data:
                values = l.strip().split(":")
                cate_dict[int(values[0])] += 1
        
        return cate_dict
    
    def preprocess(self, data):
        # pdb.set_trace()
        values = data.strip().split()
        labels = float(values[0])
        indexs = [int(v.split(':')[0]) - sum(self.feature_sizes[:i]) for i, v in enumerate(values[1:])]
        socres = [int(v.split(':')[1]) for v in values[1:]]

        # pdb.set_trace()
        return labels, torch.LongTensor(indexs), torch.FloatTensor(socres)

    def read_data(self):
        feature_sizes = self.load_category_index()
        with open(self.train, 'r') as f:
            datas = f.readlines()
        # pdb.set_trace()
        return feature_sizes, datas