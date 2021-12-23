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
    ent_vocab_path = config["Paths"]["EntVocabPath"]
    rel_vocab_path = config["Paths"]["RelVocabPath"]
    ent_id2idx_path = config["Paths"]["Entid2idxPath"]
    rel_id2idx_path = config["Paths"]["Relid2idxPath"]
    date = config["Paths"]["Date"]
    use_wiki = eval(config["Settings"]["use_wiki"])

    model_path = (
        model_path + date + "_wiki.pkl" if use_wiki else model_path + date + ".pkl"
    )
    model = torch.load(model_path, map_location="cpu")  # 読み出し

    with open(ent_vocab_path, "rb") as f:
        ent_vocab = pickle.load(f)
    with open(rel_vocab_path, "rb") as f:
        rel_vocab = pickle.load(f)
    with open(ent_id2idx_path, "rb") as f:
        ent_id2idx = pickle.load(f)
    with open(rel_id2idx_path, "rb") as f:
        rel_id2idx = pickle.load(f)

    while True:
        try:
            print("please input your question:")
            # input_question = input()
            input_question = "what does horse eat?"
            sentence = list(sentencize(input_question.lower()))[0]
            print("Input:", sentence)
            head_w, tail_w = get_entities(sentence.text)
            rel_w = get_relation(sentence.text)
            print(f"head: {head_w}, tail: {tail_w}, relation: {rel_w}")
            head_id, tail_id, rel_id = (
                ent_vocab[head_w],
                ent_vocab[tail_w],
                rel_vocab[rel_w],
            )
            head_idx, tail_idx, rel_idx = (
                ent_id2idx[head_id],
                ent_id2idx[tail_id],
                rel_id2idx[rel_id],
            )
            # print(f"head:{head}, tail: {tail}, relation: {relation}")

            # 疑問詞を特定

            if cuda.is_available():
                model.cuda()
            triple = torch.tensor([[head_idx], [tail_idx], [rel_idx]])

            # Link prediction evaluation on test set.
            evaluator = AnserPredictor(model, triple)

            if 2 <= head_id <= 7:
                target = "head"
            elif 2 <= tail_id <= 7:
                target = "tail"
            elif 2 <= rel_id <= 7:
                target = "relation"
            else:
                raise NotImplementedError

        except KeyError:
            print("Key Error. Input again")
        except NotImplementedError:
            print("can't find word asked")
        else:
            break

    evaluator.evaluate(b_size=1)
    topk_answers = evaluator.predict(pred_obj=target, topk=10)
    ent_word_list = list(ent_vocab.keys())
    rel_word_list = list(rel_vocab.keys())
    ent_id_list = [id for id, idx in sorted(ent_id2idx.items(), key=lambda x: x[1])]
    rel_id_list = [id for id, idx in sorted(rel_id2idx.items(), key=lambda x: x[1])]

    for i, answer_idx in enumerate(topk_answers):
        if target == "head" or target == "tail":
            print(i, ent_word_list[ent_id_list[answer_idx]])
        else:
            print(i, rel_word_list[rel_id_list[answer_idx]])

    # visualize(head_w, rel_w, tail_w)


if __name__ == "__main__":
    main()
