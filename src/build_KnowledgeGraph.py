from spacy.tokens import Span
from spacy.matcher import Matcher
from tqdm import tqdm
import pandas as pd
import spacy
from spacy import displacy
from src.utils.preprocess import sentencize, get_entities, get_relation


data = pd.read_csv("data/shortjokes.csv", encoding="utf-8")

KG = []


for joke in tqdm(data["Joke"]):
    for sentence in sentencize(joke):
        head, tail = get_entities(sentence.text)
        relation = get_relation(sentence.text)
        KG.append([head, relation, tail])

kg_df = pd.DataFrame(KG, columns=["head", "relation", "tail"])
kg_df.to_csv("data/knowlegegraph.csv")
