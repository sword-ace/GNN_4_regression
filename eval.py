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


def test(model):
  model_name = 'temp_model' 
  model = torch.load(os.path.join('.', model_name+'.pkl'))
  data_helper = DataHelper(mode = 'test')
  b_size = len(data_helper.label)
  print('test set total:',b_size)

  loss_func = torch.nn.MSELoss(reduction='sum')
  total_pred= 0
  correct = 0
  iter = 0
  accuracy = 0
  pre_score = []
  score = []
  for content, label, _ in data_helper.batch_iter(batch_size = b_size, num_epoch=1):
      iter +=1
      model.eval()
      pred = model(content)
      pred_sq = torch.squeeze(pred,1)
      loss = loss_func(pred_sq, label.float())
      error = mean_absolute_error(pred_sq.cpu().data, label.cpu(), multioutput = 'raw_values')
      accuracy += error
      total_pred = len(label)
      pre_score.append(pred_sq.cpu().data)
      score.append(label.cpu())

  total_pred = float(total_pred)
  accuracy = float(accuracy)
  print(torch.div(accuracy, total_pred))
  _ = result_pull(pre_score, score)
  # print('iter is:', iter)
  print('pred result:%.2f, true score:%d'%(pre_score,score))
  
  return torch.div(accuracy, total_pred)

def result_pull(pred, label):
  for i in range(len(pred[0])):
    
    print('pred result:%.2f, true score:%d'%(pred[0][i],label[0][i]))
  return None  





