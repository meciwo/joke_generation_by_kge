import configparser
from torch import cuda
import pickle

from torchkge.models.translation import TransEModel
from utils.preprocess import get_entities, get_relation, sentencize
from utils.prediction import AnserPredictor
import torch

from visualize_graph import visualize


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")
    model_path = config["Paths"]["ModelPath"]
    ent2id_path = config["Paths"]["Ent2idPath"]
    rel2id_path = config["Paths"]["Rel2idPath"]
    date = config["Paths"]["Date"]
    use_wiki = eval(config["Settings"]["use_wiki"])

    model_path = (
        model_path + date + "_wiki.pkl" if use_wiki else model_path + date + ".pkl"
    )
    model = torch.load(model_path, map_location="cpu")  # 読み出し
    with open(ent2id_path, "rb") as f:
        ent2id = pickle.load(f)
    with open(rel2id_path, "rb") as f:
        rel2id = pickle.load(f)

    while True:
        try:
            print("please input your question:")
            input_question = input()
            # input_question = "what does horse eat?"
            sentence = list(sentencize(input_question.lower()))[0]
            print("Input:", sentence)
            head_w, tail_w = get_entities(sentence.text)
            rel_w = get_relation(sentence.text)
            print(f"head: {head_w}, tail: {tail_w}, relation: {rel_w}")

            q_words = {
                "why",
                "how",
                "where",
                "who",
                "when",
                "what",
            }

            if head_w in q_words:
                target = "head"
                head_w = ""
            elif tail_w in q_words:
                target = "tail"
                tail_w = ""
            elif rel_w in q_words:
                target = "relation"
                rel_w = ""
            else:
                raise NotImplementedError

            head_idx, tail_idx, rel_idx = (
                ent2id[head_w],
                ent2id[tail_w],
                rel2id[rel_w],
            )

            # 疑問詞を特定

            if cuda.is_available():
                model.cuda()
            triple = torch.tensor([[head_idx], [tail_idx], [rel_idx]])

            # Link prediction evaluation on test set.
            evaluator = AnserPredictor(model, triple)

        except KeyError:
            print("Key Error. Input again")
        except NotImplementedError:
            print("can't find word asked")
        else:
            break

    evaluator.evaluate(b_size=1)
    topk_answers = evaluator.predict(pred_obj=target, topk=10)
    ent_list = [id for id, idx in sorted(ent2id.items(), key=lambda x: x[1])]
    rel_list = [id for id, idx in sorted(rel2id.items(), key=lambda x: x[1])]

    for i, answer_idx in enumerate(topk_answers):
        if target == "head" or target == "tail":
            print(i, ent_list[answer_idx])
        else:
            print(i, rel_list[answer_idx])

    if target == "head":
        visualize(ent_list[topk_answers[0]], rel_w, tail_w)
    elif target == "tail":
        visualize(head_w, rel_w, ent_list[topk_answers[0]])
    else:
        visualize(head_w, rel_list[topk_answers[0]], tail_w)


if __name__ == "__main__":
    main()
