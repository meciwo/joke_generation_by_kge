from torch import cuda
from torch.optim import Adam
import torch
from torch.optim.optimizer import Optimizer

from torchkge.models import TransEModel
from torchkge.models.interfaces import TranslationModel
from torchkge.sampling import BernoulliNegativeSampler, NegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from utils.datasets import load_joke_dataset, load_wiki_dataset

from tqdm.autonotebook import tqdm
import configparser
import numpy as np


def train(
    model: TransEModel,
    use_cuda: bool,
    criterion: MarginLoss,
    optimizer: Optimizer,
    dataloader: DataLoader,
    sampler: BernoulliNegativeSampler,
    iterator: tqdm,
):
    losses = []

    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            if use_cuda:
                h, t, r, n_h, n_t = h.cuda(), t.cuda(), r.cuda(), n_h.cuda(), n_t.cuda()

            # forward + backward + optimize
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            "Epoch {} | mean loss: {:.5f}".format(
                epoch + 1, running_loss / len(dataloader)
            )
        )
        losses.append(running_loss / len(dataloader))
        model.normalize_parameters()
    model.normalize_parameters()
    return model, losses


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")
    model_path = config["Paths"]["ModelPath"]
    training_log_path = config["Paths"]["TrainingLogPath"]
    date = config["Paths"]["Date"]

    use_cuda = config["Settings"]["use_cuda"] == "True"
    use_wiki = config["Settings"]["use_wiki"] == "True"
    dry_run = config["Settings"]["dry_run"] == "True"

    # Load dataset
    kg_train, _, _ = load_joke_dataset(
        "./data", valid_size=100, test_size=100, dry_run=dry_run
    )

    # Define some hyper-parameters for training
    emb_dim = int(config["Hyparas"]["emb_dim"])
    lr = float(config["Hyparas"]["lr"])
    n_epochs = int(config["Hyparas"]["n_epochs"])
    b_size = int(config["Hyparas"]["b_size"])
    margin = float(config["Hyparas"]["margin"])

    if dry_run:
        emb_dim, n_epochs = 100, 10

    ent_emb, rel_emb = None, None
    if use_wiki:
        ent_emb, rel_emb = load_wiki_dataset(kg_train)

    # Define the model and criterion
    model = TransEModel(
        emb_dim,
        kg_train.n_ent,
        kg_train.n_rel,
        ent_emb,
        rel_emb,
        dissimilarity_type="L2",
    )
    criterion = MarginLoss(margin)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=use_cuda)
    iterator = tqdm(range(n_epochs), unit="epoch")

    # Move everything to CUDA if available
    if cuda.is_available() and use_cuda:
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()
    else:
        use_cuda = False

    model, losses = train(
        model=model,
        use_cuda=use_cuda,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=dataloader,
        sampler=sampler,
        iterator=iterator,
    )

    if not dry_run:
        torch.save(model.to("cpu"), model_path + date + ".pkl")
        np.save(training_log_path + date, losses)


if __name__ == "__main__":
    main()
