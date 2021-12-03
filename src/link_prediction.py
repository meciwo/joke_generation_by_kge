from torch import cuda
from torchkge.utils.pretrained_models import load_pretrained_transe
from torchkge.utils.datasets import load_fb15k
from torchkge.evaluation import LinkPredictionEvaluator
import pickle
from utils.datasets import load_joke_dataset

_, kg_valid, kg_test = load_joke_dataset(
    './data', valid_size=100, test_size=100)

with open("./model/KG_2021_5_15.pkl", "rb") as f:
    model = pickle.load(f)  # 読み出し
if cuda.is_available():
    model.cuda()

# Link prediction evaluation on test set.
evaluator = LinkPredictionEvaluator(model, kg_test)
evaluator.evaluate(b_size=32)
evaluator.print_results()
