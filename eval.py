"""
evaluation 
"""

import torch
from model import Model
from data_helper import DataHelper
import numpy as np
import tqdm
import sys, random
import time, datetime
import os
from pmi import cal_PMI
from sklearn.metrics import accuracy_score,mean_squared_error
from laplotter import LossAccPlotter

model = train(ngram = ngram, name = name, dataset = dataset, bar=bar, drop_out=dropout, is_cuda=True, edges=edges)


def eval_test(model,data_helper, b_size=1):
  
  pre_score = []
  score = []
  iter = 0
  for content, label, _ in data_helper.batch_iter(batch_size = b_size, num_epoch=1):
      iter +=1
      model.eval()
      pred = model(content)
      pred_sq = torch.squeeze(pred,1)

      pre_score.append(pred_sq.cpu().data)
      score.append(label.cpu())

  _ = result_pull(pre_score, score)

  return None



