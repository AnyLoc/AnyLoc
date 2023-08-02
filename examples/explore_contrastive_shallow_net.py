"""
Krishna's exploration of training a shallow contrastive MLP
over CLIP embeddings for VPR
"""

import glob
import os
import random
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm, trange

from configs import BaseDatasetArgs, base_dataset_args
from dvgl_benchmark.datasets_ws import BaseDataset


class EmbeddingDataset(torch.utils.data.Dataset):
    """Skeleton dataset class to load precomputed CLIP embeddings. """

    def __init__(self, cachedir, start=0, end=-1, stride=1, positive_inds=2, num_negative_examples=1):
        # postive indices indicates the number of images before/after the current image
        # that should be used as positive examples
        self.embedding_files = natsorted(glob.glob(os.path.join(cachedir, "*.pt")))
        if end == -1:
            self.embedding_files = self.embedding_files[start::stride]
        else:
            self.embedding_files = self.embedding_files[start:end:stride]
        self.positive_inds = positive_inds
        self.dataset_size = len(self.embedding_files)
        self.embeddings = []
        for i in range(len(self.embedding_files)):
            self.embeddings.append(torch.load(self.embedding_files[i])[0])
        self.embeddings = torch.stack(self.embeddings)
        self.mean = self.embeddings.mean(0).unsqueeze(0)
        self.std = self.embeddings.std(0).unsqueeze(0)
        self.embeddings = (self.embeddings - self.mean) / (self.std)
        self.num_negative_examples = num_negative_examples

    def __getitem__(self, index):
        # Sample a positive and a negative example
        pos_inds = [i for i in range(index - self.positive_inds, index + self.positive_inds, 1)]
        pos_inds = [i for i in pos_inds if i < self.dataset_size]
        pos_ind = random.choice(pos_inds)
        neg_inds = [i for i in range(self.dataset_size) if i not in pos_inds]
        neg_ind = random.sample(neg_inds, self.num_negative_examples)
        # Return embedding, positive example, and negative example
        return (
            self.embeddings[index],
            self.embeddings[pos_ind],
            self.embeddings[neg_ind],
        )
    
    def __len__(self):
        return len(self.embedding_files)


class ShallowMLP(torch.nn.Module):
    def __init__(self, feat_in=1024, feat_out=32):
        super().__init__()
        self.linear1 = torch.nn.Linear(feat_in, 512)
        self.layernorm1 = torch.nn.LayerNorm(512)
        self.linear2 = torch.nn.Linear(512, 256)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


def compute_contrastive_loss(emb, pos, neg, temp=1.0):
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    pos = torch.nn.functional.normalize(pos, p=2, dim=-1)
    neg = torch.nn.functional.normalize(neg, p=2, dim=-1)
    sim_emb_pos = torch.nn.functional.cosine_similarity(emb, pos, dim=-1)
    sim_emb_neg = torch.nn.functional.cosine_similarity(emb.unsqueeze(1), neg, dim=-1)
    denom = torch.exp(sim_emb_neg / temp).sum(-1)
    # return - torch.log(torch.exp(sim_emb_pos / temp) / torch.exp(sim_emb_neg / temp))
    return - torch.log(torch.exp(sim_emb_pos / temp) / denom)


def compute_tsne(dataset):
    from sklearn.manifold import TSNE
    
    data = []
    for i in range(len(dataset)):
        emb, *_ = dataset[i]
        data.append(emb)
    data = torch.stack(data).cuda()
    print(data.shape)
    print("Computing t-SNE embeddings")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(data.detach().cpu().numpy())
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.show()
    plt.close("all")


def main():
    """Main method """

    cachedir = "../logs/experiments/17places/images"
    lr = 1e-3
    batch_size = 100
    num_epochs = 500
    print_every = 1000
    val_every = 10
    feat_in_dim = 1024
    feat_out_dim = 128
    num_negative_examples = 100  # num negative examples to use in contrastive loss
    temperature = 0.1

    bd_args: BaseDatasetArgs = base_dataset_args
    vg_dataset = BaseDataset(
        bd_args,
        datasets_folder="/home/krishna/Downloads/datasets_vg",
        dataset_name="17places",
        split="train",
    )
    soft_positives_per_query = vg_dataset.get_positives()
    max_num_positives = max([len(sp) for sp in soft_positives_per_query])
    # pad '-1's so that all soft positives are of the same length
    # (so they can be concatenated to a torch tensor)
    for idx, sp in enumerate(soft_positives_per_query):
        _l = len(sp)
        if _l < max_num_positives:
            pad_needed = max_num_positives - _l
            for _ in range(pad_needed):
                soft_positives_per_query[idx].append(-1)
    soft_positives_per_query = torch.LongTensor(soft_positives_per_query).cuda()

    dataset = EmbeddingDataset(
        cachedir, end=vg_dataset.database_num, positive_inds=2, num_negative_examples=num_negative_examples
    )

    net = ShallowMLP(feat_in=feat_in_dim, feat_out=feat_out_dim)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    mu = dataset.mean.cuda()
    std = dataset.mean.cuda()

    # # (Optional) Visualize t-SNE embeddings of pretrained CLIP features
    # compute_tsne(dataset)

    for e in trange(num_epochs):

        # Train loop
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_avg = 0.0
        num_steps = 0
        for idx, batch in enumerate(dataloader):
            emb_in, emb_pos, emb_neg = batch
            emb_in = (emb_in.cuda() - mu) / std
            emb_pos = (emb_pos.cuda() - mu) / std
            emb_neg = (emb_neg.cuda() - mu) / std
            proj_in = net(emb_in)
            proj_pos = net(emb_pos)
            proj_neg = net(emb_neg)
            loss = compute_contrastive_loss(proj_in, proj_pos, proj_neg, temp=temperature).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_avg += loss.item()
            num_steps += 1
        print(f"Epoch {e}; Loss: {loss_avg / num_steps}")
        
        # Validation loop
        with torch.no_grad():
            if e % val_every == 0 or e == num_epochs - 1:
                ref_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                ref_embeddings = []
                for idx, batch in enumerate(ref_dataloader):
                    emb_in, *_ = batch
                    emb_in = (emb_in.cuda() - mu) / std
                    proj = net(emb_in)
                    proj = torch.nn.functional.normalize(proj)
                    ref_embeddings.append(proj)
                ref_embeddings = torch.cat(ref_embeddings, dim=0)
                query_dataset = EmbeddingDataset(cachedir, start=vg_dataset.database_num)
                query_dataloader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
                query_embeddings = []
                for idx, batch in enumerate(query_dataloader):
                    emb_in, *_ = batch
                    emb_in = (emb_in.cuda() - mu) / std
                    proj = net(emb_in)
                    proj = torch.nn.functional.normalize(proj)
                    query_embeddings.append(proj)
                query_embeddings = torch.cat(query_embeddings, dim=0)
                
                # For each query vector, compute K-nearest reference vectors
                cos_sim_mat = torch.nn.functional.cosine_similarity(
                    query_embeddings[:, :, None], ref_embeddings.t()[None, :, :]
                )
                _, ind_top1 = cos_sim_mat.topk(k=1, dim=-1)
                _, ind_top5 = cos_sim_mat.topk(k=5, dim=-1)
                _, ind_top10 = cos_sim_mat.topk(k=10, dim=-1)
                recalls = [0, 0, 0]
                for i in range(ind_top1.shape[0]):
                    recalls[0] += torch.any(torch.isin(ind_top1[i], soft_positives_per_query[i])).float()
                    recalls[1] += torch.any(torch.isin(ind_top5[i], soft_positives_per_query[i])).float()
                    recalls[2] += torch.any(torch.isin(ind_top10[i], soft_positives_per_query[i])).float()
                recalls = [r.item() / ind_top1.shape[0] for r in recalls]
                print(f"Recalls [R@1, R@5, R@10]: {recalls}")
    
    with torch.no_grad():
        from sklearn.manifold import TSNE
        
        data = []
        for i in range(len(dataset)):
            emb, *_ = dataset[i]
            emb = (emb.cuda() - mu) / std
            proj = net(emb)
            proj = torch.nn.functional.normalize(proj)
            data.append(proj)
        data = torch.cat(data, dim=0).cuda()
        print(data.shape)
        print("Computing t-SNE embeddings")
        
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(data.detach().cpu().numpy())
        plt.scatter(tsne[:, 0], tsne[:, 1])
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
