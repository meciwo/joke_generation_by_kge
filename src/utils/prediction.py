from torch import empty
from tqdm.autonotebook import tqdm
from torchkge.utils import DataLoader


class AnserPredictor(object):

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph

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

        if use_cuda:
            dataloader = DataLoader(self.kg, batch_size=b_size,
                                    use_cuda='batch')
        else:
            dataloader = DataLoader(self.kg, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Link prediction evaluation'):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

            self.head, self.tail, _, self.relation = self.model.lp_prep_cands(
                h_idx, t_idx, r_idx)

        self.evaluated = True

        if use_cuda:
            self.head = self.head.cpu()
            self.tail = self.tail.cpu()
            self.relation = self.relation.cpu()

    def predict(self, pred_obj):
        if pred_obj == "head":
            candidate = self.tail - self.relation
        elif pred_obj == "tail":
            candidate = self.head + self.relation
        elif pred_obj == "relation":
            candidate = self.tail - self.head
        else:
            raise ValueError

        answer = calc_nearest(candidate)

        return answer
