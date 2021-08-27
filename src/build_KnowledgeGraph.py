from tqdm import tqdm
import pandas as pd
from openie import MyOpenIE
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
jokedata_path = config["Paths"]["JokeDataPath"]
kg_path = config["Paths"]["KnowlegeGraphPath"]

data = pd.read_csv(jokedata_path, encoding="utf-8")

kg = []

with MyOpenIE() as client:
    for joke in tqdm(data["Joke"]):
        ann = client.annotate(
            text=joke.lower(),
            properties={
                "openie.max_entailments_per_clause": 100,
                "openie.triple.strict": True,
            },
            simple_format=True,
        )
        for triple in ann:
            kg.append([triple["subject"], triple["relation"], triple["object"]])

kg_df = pd.DataFrame(kg, columns=["head", "relation", "tail"]).drop_duplicates()
kg_df.to_csv(kg_path)
