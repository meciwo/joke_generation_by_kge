from torch import cuda
from torchkge.utils.pretrained_models import load_pretrained_transe
from torchkge.utils.datasets import load_fb15k
from torchkge.evaluation import LinkPredictionEvaluator
import pickle
from utils.datasets import to_knowledge_graph
from utils.preprocess import get_entities, get_relation, sentencize


with open("./model/KG_2020_5_15.pkl", "rb") as f:
    model = pickle.load(f)  # 読み出し
with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

input_question = "What is your favorite food?"
sentence = list(sentencize(input_question))[0]
head, tail = get_entities(sentence.text)
relation = get_relation(sentence.text)

head, tail, relation = vocab[head], vocab[tail], vocab[relation]

if cuda.is_available():
    model.cuda()
triple = to_knowledge_graph(head, tail, relation)

# Link prediction evaluation on test set.
evaluator = LinkPredictionEvaluator(model, triple)
evaluator.evaluate(b_size=1)
evaluator.print_results()
