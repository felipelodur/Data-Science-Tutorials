# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:05:42 2017

@author: lipec
"""

import scipy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import numpy as np


def recommendation(model, data, user_ids):
    
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices] #compressed sparse row
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)] 

        print("Recommended Movies:")
        for x in top_items[:10]:
            print("   %s" % x)



def main ( ):
    
    data = fetch_movielens(min_rating=4.0)
             
    # Warp Model: Weighted Approximate-Rank Pairwise
    # Gradient Descent Algorithm to find the weights and improve predictions
    # Content + Collaboration (Hybrid Rec Sys)
    model = LightFM(loss='warp')
    model.fit(data['train'], epochs=30, num_threads=2)
    
    recommendation(model, data, [3, 25, 450])
