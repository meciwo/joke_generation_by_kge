from torch import cuda
from torch.optim import Adam
import torch

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from utils.datasets import load_joke_dataset

from tqdm.autonotebook import tqdm
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
model_path = config["Paths"]["ModelPath"]

use_cuda = True
# Load dataset
kg_train, _, _ = load_joke_dataset("./data", valid_size=100, test_size=100)

# Define some hyper-parameters for training
emb_dim = int(config["Hyparas"]["emb_dim"])
lr = float(config["Hyparas"]["lr"])
n_epochs = int(config["Hyparas"]["n_epochs"])
b_size = int(config["Hyparas"]["b_size"])
margin = float(config["Hyparas"]["margin"])

# Define the model and criterion
model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type="L2")
criterion = MarginLoss(margin)

# Move everything to CUDA if available
if cuda.is_available() and use_cuda:
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()
    use_cuda = True

# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=use_cuda)

iterator = tqdm(range(n_epochs), unit="epoch")
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, r, t = batch[0], batch[1], batch[2]
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
        "Epoch {} | mean loss: {:.5f}".format(epoch + 1, running_loss / len(dataloader))
    )

model.normalize_parameters()
torch.save(model.to("cpu"), model_path)
