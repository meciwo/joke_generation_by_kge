import configparser
from os.path import exists
from typing import List, Tuple
from pandas import concat, DataFrame, merge, read_csv
import torch
from torch.nn.modules.sparse import Embedding

from torchkge.data_structures import KnowledgeGraph
import numpy as np
from torchkge.utils.modeling import init_embedding
from wikipedia2vec import Wikipedia2Vec
import pickle


config = configparser.ConfigParser()
config.read("config.ini")
wiki_vec_path = config["Paths"]["WikiVecPath"]
ent_vocab_path = config["Paths"]["EntVocabPath"]
rel_vocab_path = config["Paths"]["RelVocabPath"]


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


def load_wiki_dataset(
    dataset: KnowledgeGraph, emb_dim: int = 100
) -> Tuple[Embedding, Embedding]:

    with open(ent_vocab_path, "rb") as f:
        ent_vocab = pickle.load(f)
    with open(rel_vocab_path, "rb") as f:
        rel_vocab = pickle.load(f)

    ent_word = list(ent_vocab.keys())
    rel_word = list(rel_vocab.keys())

    ent2ix = dataset.ent2ix
    rel2ix = dataset.rel2ix
    n_ent = dataset.n_ent
    n_rel = dataset.n_rel

    ent_emb = init_embedding(n_ent, emb_dim)
    rel_emb = init_embedding(n_rel, emb_dim)

    wiki2vec = Wikipedia2Vec.load(wiki_vec_path)

    ent_weight, rel_weight = [torch.rand(emb_dim) for _ in range(n_ent)], [
        torch.rand(emb_dim) for _ in range(n_rel)
    ]
    for ent, idx in ent2ix.items():
        ent_list = split_phrase_to_word(ent_word[ent])
        ent_vec = []
        for _ent in ent_list:
            try:
                ent_vec.append(torch.Tensor(wiki2vec.get_word_vector(_ent)))
            except KeyError:
                continue
        if ent_vec != []:
            ent_weight[idx] = torch.stack(ent_vec).sum(dim=0)

    for rel, idx in rel2ix.items():
        rel_list = split_phrase_to_word(rel_word[rel])

        rel_vec = []
        for _rel in rel_list:
            try:
                rel_vec.append(torch.Tensor(wiki2vec.get_word_vector(_rel)))
            except KeyError:
                continue
        if rel_vec != []:
            rel_weight[idx] = torch.stack(rel_vec).sum(dim=0)

    ent_emb.weight = torch.nn.Parameter(torch.stack(ent_weight))
    rel_emb.weight = torch.nn.Parameter(torch.stack(rel_weight))

    return ent_emb, rel_emb


def split_phrase_to_word(phrase: str) -> List[str]:
    if type(phrase) != str:
        return ""
    return phrase.split()
