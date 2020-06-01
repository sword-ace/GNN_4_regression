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

NUM_ITER_EVAL = 150  #100
EARLY_STOP_EPOCH = 250  #250


"""## add window size create connect matrix ##"""

def edges_mapping(vocab_len, content, ngram):
  count = 1
  mapping = np.zeros(shape = (vocab_len, vocab_len), dtype = np.int32)
  for doc in content:
    for i , src in enumerate(doc):
      for dst_id in range(max(0, i-ngram), min(len(doc), i+ngram+1)):
        dst = doc(dst_id)
        if mapping[src, dst] == 0:
          mapping[src, dst] = count
          count +=1
#add self node connection
  for word in range(vocab_len):
    mapping[word, word] = count
    count += 1
  return count, mapping

def get_time_dif(start_time):
  end_time = time.time()
  time_dif = end_time - start_time
  return datetime.timedelta(seconds = int(round(time_dif)))

"""## plot ##"""



def dev(model):
  data_helper = DataHelper(mode='dev')
  total_pred = 0
  correct = 0
  accuracy = 0
  b_size = len(data_helper.label)
  print('*'*100)
  print('dev set total:',b_size)
  loss_func = torch.nn.MSELoss(reduction='sum') ##reduction='sum'
  loss_mae = torch.nn.L1Loss(reduction='sum') ##reduction='sum'

  iter = 0
  total_loss = 0
  for content, label , _ in data_helper.batch_iter(batch_size = b_size, num_epoch=1):
      iter +=1
      model.eval()
      ## the original version is for classification task here is just regression:
      # logits = model(content)

      ##need modify the regression task will minimize the mse error
      # pred = torch.argmax(logits, dim =1)
      pred = model(content)
      pred_sq = torch.squeeze(pred,1)
      loss = loss_func(pred_sq.cpu().data, label.cpu())
     #------------------------------------------------#
      error = loss_mae(pred_sq.cpu().data, label.cpu())
      # error = mean_absolute_error(pred_sq.cpu().data, label.cpu())
      accuracy += error
      # correct_pred = torch.sum(pred == label)

      # correct += correct_pred
      # count total doc number
      total_pred = len(label)

  total_pred = float(total_pred)
  # correct = correct.float()
  accuracy = float(accuracy)
 
  
  #return the overall accuracy   torch.div(accuracy, total_pred)
  return (accuracy/total_pred), (float(loss)/total_pred)

def test(model):
  model_name = 'temp_model' 
  model = torch.load(os.path.join('.', model_name+'.pkl'))
  data_helper = DataHelper(mode = 'test')
  b_size = len(data_helper.label)
  print('test set total:',b_size)

  loss_func = torch.nn.MSELoss(reduction='sum') ##reduction='sum'

  total_pred= 0
  correct = 0
  iter = 0
  accuracy = 0
  pre_score = []
  score = []
  for content, label, _ in data_helper.batch_iter(batch_size = b_size, num_epoch=1):
      iter +=1
      model.eval()
      # logits = model(content)
      # pred = torch.argmax(logtist,dim=1)
      pred = model(content)
      pred_sq = torch.squeeze(pred,1)
      loss = loss_func(pred_sq, label.float())
      # pred = torch.argmax(logits, dim =1)
      #--------------------------------------------------#
      # error = mean_squared_error(pred_sq.cpu().data, label.cpu())
      error = mean_absolute_error(pred_sq.cpu().data, label.cpu(), multioutput = 'raw_values')
      accuracy += error
      # correct_pred = torch.sum(pred==label)
      # correct += correct_pred
      total_pred = len(label)
      pre_score.append(pred_sq.cpu().data)
      score.append(label.cpu())

  total_pred = float(total_pred)
  # correct = correct.float()
  accuracy = float(accuracy)
  # pre_score = float(pred_sq.cpu().data)
  # score = 
  print(torch.div(accuracy, total_pred))
  _ = result_pull(pre_score, score)
  # print('iter is:', iter)
  print('pred result:%.2f, true score:%d'%(pre_score,score))
  # return torch.div(correct, total_pred).to('cpu')
  
  return torch.div(accuracy, total_pred)

def result_pull(pred, label):
  for i in range(len(pred[0])):
    
    print('pred result:%.2f, true score:%d'%(pred[0][i],label[0][i]))
  return None  


def train(ngram, name, bar, drop_out, dataset, is_cuda= False, edges=False):
  plotter = LossAccPlotter(title="This is an example plot",
                         save_to_filepath="/tmp/my_plot.png",
                         show_regressions=True,
                         show_averages=True,
                         show_loss_plot=True,
                         show_acc_plot=True,
                         show_plot_window=False,
                         x_label="Epoch")
  
  print('load data helper.')
  data_helper = DataHelper(mode = 'train')
  b_size = len(data_helper.label)
  print('*'*100)
  print('train set total:', b_size)

  if os.path.exists(os.path.join('.', name+ '.pkl')) and name != 'temp_model' :
    print('load model from file')
    model = torch.load(os.path.join('.', name+ '.pkl'))

  else:
    print('new model')
    if name == 'temp_model':
      name == 'temp_model'
    
    edges_weights, edges_mappings, count = cal_PMI()

## -----------------************************** import the datahelper class to get the vocab-5 doc*****************------------------------
 
  ## class_num = len(data_helper.labels_str) is changed, consider just a score
    model = Model(class_num = data_helper.labels_str, hidden_size_node=200,
                  vocab = data_helper.vocab, n_gram = ngram, drop_out = drop_out, edges_matrix = edges_mappings, edges_num = count,
                  trainable_edges = edges, pmi = edges_weights, cuda = is_cuda
                      )
  ### --------------------------------------- ###
    print(model)

    if is_cuda:
      print('cuda')
      model.cuda()
    loss_func = torch.nn.MSELoss()
    loss_mae = torch.nn.L1Loss(reduction='sum')
    optim = torch.optim.Adam(model.parameters(), weight_decay= 1e-3)
    iter = 0
    if bar:
      pbar = tqdm.tqdm(total = NUM_ITER_EVAL)

    best_acc = 4.4 #0.0
    last_best_epoch = 0
    start_time = time.time()
    total_loss= 0.0
    total_correct = 0
    total = 0
    accuracy = 0.0
    num_epoch = 700
    weight_decays = 1e-4
    for content, label, epoch in data_helper.batch_iter(batch_size = 32,  num_epoch = num_epoch):
      improved = ''
      model.train()
      pred = model(content)
      pred_sq = torch.squeeze(pred,1)
      l2_reg=0.5*weight_decays*(model.seq_edge_w.weight.to
                          ('cpu').detach().numpy()**2).sum()
    
      loss = loss_func(pred_sq, label.float()) + l2_reg
     
      #-------------------------------------------#
      error = loss_mae(pred_sq.cpu().data, label.cpu())
      accuracy += error 
      total += len(pred)  ##batch size = len(label)
      total_loss += (loss.item()*len(pred))
      total_correct += loss.item()
      optim.zero_grad()
      loss.backward()
      optim.step()

      iter +=1
      if bar:
            pbar.update()

      if iter% NUM_ITER_EVAL == 0:
          if bar:
            pbar.close()

          val_acc, val_loss = dev(model)
          
          
          if val_acc < best_acc:
              best_acc = val_acc
              last_best_epoch = epoch
              improved = '* '
              torch.save(model, name + '.pkl')
          
          msg = 'Epoch: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Train Error: {6:>7.2}' \
                + 'Val Acc: {2:>7.2}, Time: {3}{4}, val error:{7:>7.2}' \
                  # + ' Time: {5} {6}'

          print(msg.format(epoch, iter, val_acc, get_time_dif(start_time), improved, total_correct/(NUM_ITER_EVAL),
                            float(accuracy) / float(total), val_loss))
          
          plotter.add_values(epoch,
                       loss_train=total_correct/(NUM_ITER_EVAL), acc_train=float(accuracy) / float(total),
                        
                       loss_val=val_loss, acc_val=best_acc)


          total_loss = 0.0
          total_correct = 0
          accuracy = 0.0
          total = 0
          if bar:
              pbar = tqdm.tqdm(total=NUM_ITER_EVAL)

      plotter.block()
  return name

"""the function to compute pmi weights """

def cal_PMI( window_size= 20):
    helper = DataHelper(mode="train")
    content, _ = helper.get_content()
    pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    word_count =np.zeros(len(helper.vocab), dtype=int)
    
    for sentence in content:
        sentence = sentence.split(' ')
        for i, word in enumerate(sentence):
            try:
                word_count[helper.d[word]] += 1
            except KeyError:
                continue
            start_index = max(0, i - window_size)
            end_index = min(len(sentence), i + window_size)
            for j in range(start_index, end_index):
                if i == j:
                    continue
                else:
                    target_word = sentence[j]
                    try:
                        pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                    except KeyError:
                        continue
        
    total_count = np.sum(word_count)
    word_count = word_count / total_count
    pair_count_matrix = pair_count_matrix / total_count
    pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            pmi_matrix[i, j] = np.log(
                pair_count_matrix[i, j] / (word_count[i] * word_count[j]) 
            )
            if pmi_matrix[i, j] <= 0:
              continue

            
    
    pmi_matrix = np.nan_to_num(pmi_matrix)
    
    pmi_matrix = np.maximum(pmi_matrix, 0.0)

    edges_weights = [0.0]
    count = 1
    edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            if pmi_matrix[i, j] != 0:
                edges_weights.append(pmi_matrix[i, j])
                edges_mappings[i, j] = count
                count += 1

    edges_weights = np.array(edges_weights)

    edges_weights = edges_weights.reshape(-1, 1)
    # print(edges_weights.shape)
    edges_weights = torch.Tensor(edges_weights)
    
    return edges_weights, edges_mappings, count

ngram = 8
name = 'temp_model'
bar = True
dropout = 0.5
dataset = 'depress'
edges = True
rand = 20
max_length=4600 #the max len for each doc
SEED = rand

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


model = train(ngram = ngram, name = name, dataset = dataset, bar=bar, drop_out=dropout, is_cuda=True, edges=edges)


#the result by testing the distribution of each word

def edges_mapping1(vocab_len, helper, ngram):
    # helper = DataHelper(mode="train")
    content, label = helper.get_content()
    

    pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    word_count =np.zeros(len(helper.vocab), dtype=int)
    
        
    for sentence in content:
        sentence = sentence.split(' ')
        for i, word in enumerate(sentence):
            try:
                word_count[helper.d[word]] += 1
            except KeyError:
                continue
            start_index = max(0, i - ngram)
            end_index = min(len(sentence), i + ngram)
            for j in range(start_index, end_index):
                if i == j:
                    continue
                else:
                    target_word = sentence[j]
                    try:
                        pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                    except KeyError:
                        continue
        
    total_count = np.sum(word_count)
    word_count = word_count / total_count
    pair_count_matrix = pair_count_matrix / total_count
    # print(pair_count_matrix)
    pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            pmi_matrix[i, j] = np.log(
                pair_count_matrix[i, j] / (word_count[i] * word_count[j]) 
            )
    
    pmi_matrix = np.nan_to_num(pmi_matrix)
    
    pmi_matrix = np.maximum(pmi_matrix, 0.0)

    edges_weights = [0.0]
    count = 1
    edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            if pmi_matrix[i, j] != 0:
                edges_mappings[i, j] = count
                count += 1

    
    return count, edges_mappings

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



