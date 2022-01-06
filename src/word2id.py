import csv
import pickle
import configparser

from utils.datasets import load_joke_dataset

config = configparser.ConfigParser()
config.read("config.ini")
kg_path = config["Paths"]["KnowlegeGraphPath"]
train_path = config["Paths"]["TrainDataPath"]
ent2id_path = config["Paths"]["Ent2idPath"]
rel2id_path = config["Paths"]["Rel2idPath"]

"""
with open(kg_path) as f:
    reader = csv.reader(f)
    ke = [row for row in reader]

ent_word2id, rel_word2id = {}, {}
q_words = {
    "<unk>": 0,
    "": 1,
    "why": 2,
    "how": 3,
    "where": 4,
    "who": 5,
    "when": 6,
    "what": 7,
}
ent_word2id.update(q_words)
rel_word2id.update(q_words)
cnt = 8


with open(train_path, "w") as f:

    for triple in ke:
        head, relation, tail = triple.split()
        try:
            assert type(head) == str and type(relation) == str and type(tail) == str
        except AssertionError:
            print(head, relation, tail)
        f.write(" ".join([str(head), str(relation), str(tail)]) + "\n")

"""


kg_train, _, _ = load_joke_dataset("./data", valid_size=0, test_size=0)

ent2id, rel2id = kg_train.ent2ix, kg_train.rel2ix

with open(ent2id_path, "wb") as f:
    pickle.dump(ent2id, f)

with open(rel2id_path, "wb") as f:
    pickle.dump(rel2id, f)
