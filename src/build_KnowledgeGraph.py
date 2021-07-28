from tqdm import tqdm
import pandas as pd
from openie import StanfordOpenIE


data = pd.read_csv("data/shortjokes.csv", encoding="utf-8")

KG = []

with StanfordOpenIE() as client:
    for joke in tqdm(data["Joke"]):
        for triple in client.annotate(joke.lower()):
            KG.append([triple["subject"], triple["relation"], triple["object"]])

kg_df = pd.DataFrame(KG, columns=["head", "relation", "tail"])
kg_df.to_csv("data/knowlegegraph_2.csv")
