from torch import cuda
from torchkge.utils.pretrained_models import load_pretrained_transe
from torchkge.utils.datasets import load_fb15k
from torchkge.evaluation import LinkPredictionEvaluator
import pickle
from utils.datasets import load_joke_dataset


# sentence = ""
# head, tail, relation = extract(sentence)
# knowledgeGraph = to_knowledgegraph(head, tail, relation)

with open("./model/KG_2020_5_15.pkl", "rb") as f:
    model = pickle.load(f)  # 読み出し
if cuda.is_available():
    model.cuda()

# Link prediction evaluation on test set.
evaluator = LinkPredictionEvaluator(model, knowledgeGraoh)
evaluator.evaluate(b_size=1)
evaluator.print_results()
