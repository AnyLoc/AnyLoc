"""
Clustering of CLIP descriptors with vpr residuals followed by closest retrieval.
Uses fast-kmeans library to cluster.
"""

import os
import cv2
import fast_pytorch_kmeans as fpk
import numpy as np
import torch
from natsort import natsorted

import joblib

def load_descriptors(descriptor_cache_path):
    """
    Load descriptors and return torch tensor
    """
    all_pts = natsorted(os.listdir(descriptor_cache_path))
    feature_size = torch.load(os.path.join(descriptor_cache_path,all_pts[0]))

    descriptors = torch.zeros((len(all_pts),feature_size.shape[1]))

    for idx in range(len(all_pts)):
        cur_descriptor = torch.load(os.path.join(descriptor_cache_path,all_pts[idx]))
        cur_descriptor = torch.nn.functional.normalize(cur_descriptor,p=2) #cur_descriptor/torch.sqrt(cur_descriptor**2)#
        descriptors[idx] = cur_descriptor[0]
    
    return descriptors        

def get_enhanced_residual_vector(descriptors,cluster_centroids):
    """
    Get descriptors distance from cluster centers and concat them to original descriptor
    """
    num_desc = descriptors.shape[0]
    desc_dim = descriptors.shape[1]

    num_clusters = cluster_centroids.shape[0]
    residuals = torch.zeros(num_desc,(desc_dim*(num_clusters)))

    # residuals[:,:desc_dim] = torch.nn.functional.normalize(descriptors,p=2.0,dim=-1)

    # slower implementation but uses lesser memory for large vector dims
    for c in range(num_clusters):
        cur_residuals = descriptors - cluster_centroids[c]
        residuals[:,((c)*desc_dim):((c+1)*desc_dim)] = torch.nn.functional.normalize(cur_residuals,p=2.0)
    
    residuals = torch.nn.functional.normalize(residuals,p=2.0,dim=-1)

    return residuals

def retrieve_db(q,db_desc,k=5):
    """
    Given query vector, return the closest matching database vectors
    """
    sim_matrix = q @ db_desc.T

    return torch.topk(sim_matrix,k,dim=1)

def calc_acc(q_pred_list, gt_list):
    """
    Calc accuracy of correct descriptors
    """

    print("Prediction list len : {}, GT list len : {}".format(len(q_pred_list),len(gt_list)))

    ctr = 0
    for i in range(len(q_pred_list)):
        break_flag=0
        for j in range(len(q_pred_list[i])):
            
            if q_pred_list[i][j] in gt_list[i] and break_flag==0:
                break_flag=1 #set 1 to avoid adding more 
                ctr = ctr+1
                
    ctr = ctr/len(q_pred_list)

    return ctr

if __name__=="__main__":

    database = load_descriptors("/home/jay/Downloads/images/database_new/")
    query = load_descriptors("/home/jay/Downloads/images/query_new/")

    pos_seq = joblib.load("/home/jay/Downloads/st_lucia_test_info.gz")['Soft-Positives']

    print("Num of database images : {}, Num of query images : {}".format(database.shape[0],query.shape[0]))
    print("Embedding dimension :", database.shape[1])

    cluster_size = [4,8,16,32,64]
    recalls = [1,5,10,15,20]

    for k in recalls:
        print("-------------------------Recall @ {}-----------------------------".format(k))

        for num_cluster in cluster_size:
            
            print("---------------------------Num Clusters is {}---------------------------".format(num_cluster))
            kmeans = fpk.KMeans(n_clusters=num_cluster, mode='cosine', verbose=1)
            labels = kmeans.fit_predict(database)

            print("Number of clusters",kmeans.centroids.shape[0])

            enhanced_db = get_enhanced_residual_vector(database,kmeans.centroids)
            enhanced_q = get_enhanced_residual_vector(query,kmeans.centroids)
            print("Residual vector shape : ", enhanced_db.shape)

            # best_retrievals = retrieve_db(query,database,k) #as a baseline
            best_retrievals = retrieve_db(enhanced_q,enhanced_db,k)

            # print(best_retrievals.indices,pos_seq)
            # print(best_retrievals.indices.tolist()[0])

            acc = calc_acc(best_retrievals.indices.tolist(),pos_seq)
            print("accuracy is:", acc)