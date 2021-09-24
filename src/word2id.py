import pandas as pd
import pickle
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
kg_path = config["Paths"]["KnowlegeGraphPath"]
train_path = config["Paths"]["TrainDataPath"]
vocab_path = config["Paths"]["VocabPath"]

ke = pd.read_csv(kg_path)
word2id = {}
word2id["<unk>"] = 0
word2id[""] = 1
q_words = {"why": 2, "how": 3, "where": 4, "who": 5, "when": 6, "what": 7}
word2id.update(q_words)
cnt = 8
types = ["head", "relation", "tail"]
for type_ in types:
    for word in ke[type_].values:
        if word not in word2id:
            word2id[word] = cnt
            cnt += 1

with open(train_path, "w") as f:

    for head, relation, tail in ke[types].values:
        f.write(
            str(word2id[head])
            + " "
            + str(word2id[relation])
            + " "
            + str(word2id[tail])
            + "\n"
        )

with open(vocab_path, "wb") as f:
    pickle.dump(word2id, f)  # 保存
