import pandas as pd
import pickle
import configparser

from utils.datasets import load_joke_dataset

config = configparser.ConfigParser()
config.read("config.ini")
kg_path = config["Paths"]["KnowlegeGraphPath"]
train_path = config["Paths"]["TrainDataPath"]
ent_vocab_path = config["Paths"]["EntVocabPath"]
rel_vocab_path = config["Paths"]["RelVocabPath"]
ent_id2idx_path = config["Paths"]["Entid2idxPath"]
rel_id2idx_path = config["Paths"]["Relid2idxPath"]

ke = pd.read_csv(kg_path)
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
types = ["head", "relation", "tail"]

for word in list(ke["head"].values) + list(ke["tail"].values):
    if word not in ent_word2id:
        ent_word2id[word] = cnt
        cnt += 1

cnt = 8
for word in ke["relation"].values:
    if word not in rel_word2id:
        rel_word2id[word] = cnt
        cnt += 1

print(f"ent vocab size:{len(ent_word2id)}")
print(f"rel vacab size:{len(rel_word2id)}")


with open(train_path, "w") as f:

    for head, relation, tail in ke[types].values:
        f.write(
            str(ent_word2id[head])
            + " "
            + str(rel_word2id[relation])
            + " "
            + str(ent_word2id[tail])
            + "\n"
        )

with open(ent_vocab_path, "wb") as f:
    pickle.dump(ent_word2id, f)  # 保存

with open(rel_vocab_path, "wb") as f:
    pickle.dump(rel_word2id, f)  # 保存

kg_train, _, _ = load_joke_dataset("./data", valid_size=100, test_size=100)

ent_id2idx, rel_id2idx = kg_train.ent2ix, kg_train.rel2ix

with open(ent_id2idx_path, "wb") as f:
    pickle.dump(ent_id2idx, f)

with open(rel_id2idx_path, "wb") as f:
    pickle.dump(rel_id2idx, f)
