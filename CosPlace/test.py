
import os
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as T


# Compute R@1, R@5, R@10, R@20 - default if args.recall_values = None
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module, eval_dis_ds: Dataset=None) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            if args.backbone == "ViT":  # TODO: Only 224, 224 supported!
                descriptors = model(T.resize(images, (224, 224)).to(args.device))
                descriptors: torch.Tensor = descriptors.pooler_output
            else:
                descriptors: torch.Tensor = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            if args.backbone == "ViT":  # TODO: Only 224, 224 supported!
                descriptors = model(T.resize(images, (224, 224)).to(args.device))
                descriptors: torch.Tensor = descriptors.pooler_output
            else:
                descriptors: torch.Tensor = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    print(f"Database descriptors shape: {database_descriptors.shape}")
    print(f"Query descriptors shape: {queries_descriptors.shape}")
    if args.save_descs is not None:
        save_f = os.path.realpath(os.path.expanduser(args.save_descs))
        save_f = f"{save_f}.npy"
        np.save(save_f, all_descriptors)
        print(f"Descriptors saved to {save_f}")

    #VPAir stuff
    if eval_dis_ds is not None:

        with torch.no_grad():
            logging.debug("Extracting distractor database descriptors for evaluation/testing")
            dis_database_subset_ds = Subset(eval_dis_ds, list(range(eval_dis_ds.database_num)))
            dis_database_dataloader = DataLoader(dataset=dis_database_subset_ds, num_workers=args.num_workers,
                                            batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
            all_dis_descriptors = np.empty((len(eval_dis_ds), args.fc_output_dim), dtype="float32")
            for images, indices in tqdm(dis_database_dataloader, ncols=100):
                if args.backbone == "ViT":  # TODO: Only 224, 224 supported!
                    dis_descriptors = model(T.resize(images, (224, 224)).to(args.device))
                    dis_descriptors: torch.Tensor = dis_descriptors.pooler_output
                else:
                    dis_descriptors: torch.Tensor = model(images.to(args.device))
                dis_descriptors = dis_descriptors.cpu().numpy()
                all_dis_descriptors[indices.numpy(), :] = dis_descriptors
        combined_database_descriptors = np.concatenate((database_descriptors,all_dis_descriptors),0)
        database_descriptors = combined_database_descriptors
        print(f"Database with distractors shape: {database_descriptors.shape}")

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    recall_values = args.recall_values or RECALL_VALUES
    _, predictions = faiss_index.search(queries_descriptors, max(recall_values))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(recall_values))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
    return recalls, recalls_str
