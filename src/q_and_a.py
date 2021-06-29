from torch import cuda
import pickle
from utils.datasets import to_knowledge_graph
from utils.preprocess import get_entities, get_relation, sentencize
from utils.prediction import AnserPredictor


with open("./model/KG_2020_5_15.pkl", "rb") as f:
    model = pickle.load(f)  # 読み出し
with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
print(vocab["what"])
input_question = "I have a pen"
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
triple = to_knowledge_graph(head, tail, relation)

# Link prediction evaluation on test set.
evaluator = AnserPredictor(model, triple)


evaluator.evaluate(b_size=1)
topk_answers = evaluator.predict("tail", topk=5)
words = list(vocab.keys())
for answer in topk_answers:
    print(words[answer])
