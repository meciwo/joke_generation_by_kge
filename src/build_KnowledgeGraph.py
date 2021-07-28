from tqdm import tqdm
import pandas as pd
from openie import StanfordOpenIE
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
jokedata_path = config["Paths"]["JokeDataPath"]
kg_path = config["Paths"]["KnowlegeGraphPath"]

data = pd.read_csv(jokedata_path, encoding="utf-8")

KG = []

with StanfordOpenIE() as client:
    for joke in tqdm(data["Joke"]):
        for triple in client.annotate(joke.lower()):
            KG.append([triple["subject"], triple["relation"], triple["object"]])

kg_df = pd.DataFrame(KG, columns=["head", "relation", "tail"])
kg_df.to_csv(kg_path)
