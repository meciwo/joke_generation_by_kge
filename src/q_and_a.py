import configparser
from torch import cuda
import pickle

from torchkge.models.translation import TransEModel
from utils.preprocess import get_entities, get_relation, sentencize
from utils.prediction import AnserPredictor
import torch

config = configparser.ConfigParser()
config.read("config.ini")
model_path = config["Paths"]["ModelPath"]
vocab_path = config["Paths"]["VocabPath"]

model = torch.load(model_path, map_location="cpu")  # 読み出し
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
input_question = "what does the word china mean in chinese?"
sentence = list(sentencize(input_question.lower()))[0]
print("Input:", sentence)
head, tail = get_entities(sentence.text)
relation = get_relation(sentence.text)
print(f"head: {head}, tail: {tail}, relation: {relation}")
head, tail, relation = vocab[head], vocab[tail], vocab[relation]
print(f"head:{head}, tail: {tail}, relation: {relation}")


# 疑問詞を特定

if cuda.is_available():
    model.cuda()
triple = torch.tensor([[head], [tail], [relation]])

# Link prediction evaluation on test set.
evaluator = AnserPredictor(model, triple)

if 2 <= head <= 7:
    target = "head"
elif 2 <= tail <= 7:
    target = "tail"
elif 2 <= relation <= 7:
    target = "relation"
else:
    raise NotImplementedError

evaluator.evaluate(b_size=1)
topk_answers = evaluator.predict(pred_obj=target, topk=5)
words = list(vocab.keys())
for i, answer in enumerate(topk_answers):
    print(i, words[answer])
