import torch


class AnserPredictor(object):
    def __init__(self, model, triple):
        self.model = model
        self.triple = triple

        self.head = None
        self.tail = None
        self.relation = None

        self.evaluated = False

    def evaluate(self, b_size, verbose=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.

        """
        use_cuda = next(self.model.parameters()).is_cuda

        h_idx, t_idx, r_idx = self.triple
        self.head, self.tail, _, self.relation = self.model.lp_prep_cands(
            h_idx, t_idx, r_idx
        )

        self.evaluated = True

        if use_cuda:
            self.head = self.head.cpu()
            self.tail = self.tail.cpu()
            self.relation = self.relation.cpu()

    def predict(self, pred_obj, topk):
        if pred_obj == "head":
            candidate = self.tail - self.relation
        elif pred_obj == "tail":
            candidate = self.head + self.relation
        elif pred_obj == "relation":
            candidate = self.tail - self.head
        else:
            raise ValueError

        answer = self.calc_nearest(candidate, topk=topk, pred_obj=pred_obj)

        return answer

    def calc_nearest(self, candidate, topk, pred_obj):
        ent_emb, rel_emb = self.model.get_embeddings()
        if pred_obj == ("head" or "tail"):
            return torch.argsort(torch.cdist(ent_emb, candidate), dim=0)[
                :topk
            ].flatten()
        else:
            return torch.argsort(torch.cdist(rel_emb, candidate), dim=0)[
                :topk
            ].flatten()
