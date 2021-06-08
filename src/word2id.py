import pandas as pd
from collections import defaultdict
ke = pd.read_csv("data/knowlegegraph.csv")
word2id = defaultdict(int)


cnt = 0
types = ["head", "relation", "tail"]
for type_ in types:
    for word in ke[type_].values:
        word2id[word] = cnt
        cnt += 1

with open("data/train.txt", "w") as f:

    for head, relation, tail in ke[types].values:
        f.write(str(word2id[head])+" "+str(word2id[tail]) +
                " "+str(word2id[relation])+"\n")
