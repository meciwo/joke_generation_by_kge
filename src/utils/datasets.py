from os.path import exists
from pandas import concat, DataFrame, merge, read_csv

from torchkge.data_structures import KnowledgeGraph
import numpy as np


def load_joke_dataset(data_path, valid_size, test_size, dry_run: bool = False):
    train_path = data_path + "/train2id.txt"
    valid_path = data_path + "/valid2id.txt"
    test_path = data_path + "/test2id.txt"
    df1 = read_csv(
        train_path,
        sep=" ",
        header=None,
        names=["from", "rel", "to"],
        nrows=500 if dry_run else None,
    )
    if exists(valid_path):
        df2 = read_csv(valid_path, sep=" ", header=None, names=["from", "rel", "to"])
    else:
        df2 = DataFrame([], columns=["from", "rel", "to"])
    if exists(test_path):
        df3 = read_csv(test_path, sep=" ", header=None, names=["from", "rel", "to"])
    else:
        df3 = DataFrame([], columns=["from", "rel", "to"])
    df = concat([df1, df2, df3])

    kg = KnowledgeGraph(df)
    train, valid, test = kg.split_kg(
        sizes=(len(df1) - valid_size - test_size, valid_size, test_size)
    )
    return train, valid, test


def to_knowledge_graph(head, relation, tail):
    df = DataFrame(
        np.reshape([head, relation, tail], [1, 3]), columns=["from", "rel", "to"]
    )
    return KnowledgeGraph(df)
