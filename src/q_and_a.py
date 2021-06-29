from torch import cuda
import pickle
from utils.preprocess import get_entities, get_relation, sentencize
from utils.prediction import AnserPredictor
import torch


with open("./model/KG_2020_5_15.pkl", "rb") as f:
    model = pickle.load(f)  # 読み出し
with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
input_question = "What did I make?"
sentence = list(sentencize(input_question.lower()))[0]
print(sentence)
head, tail = get_entities(sentence.text)
relation = get_relation(sentence.text)
print(f'head: {head}, tail: {tail}, relation: {relation}')
head, tail, relation = vocab[head], vocab[tail], vocab[relation]
print(f'head:{head}, tail: {tail}, relation: {relation}')


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
for answer in topk_answers:
    print(words[answer])
