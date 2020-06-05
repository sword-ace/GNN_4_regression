import torch
from model import Model
from data_helper import DataHelper
import numpy as np
from pmi import cal_PMI
import matplotlib.pyplot as plt
import networkx as nx
from absl import flags
from sklearn.manifold import TSNE
import matplotlib
import sys
sys.path.insert(0, './')

# !apt install libgraphviz-dev
# !pip install pygraphviz

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout

#the result by testing the distribution of each word
def edges_dist(vocab_len, helper, ngram):
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
            pmi_matrix[i, j] = np.log(pair_count_matrix[i, j] / (word_count[i] * word_count[j]))
    
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

## all words in a doc
core_words = []
data_helper = DataHelper('test')
for i in range(len(data_helper.vocab)):
    word = data_helper.vocab[i]
    core_words.append(word)


def graph_eval(core_words):
    
    # print('load model from file.')
    data_helper = DataHelper('test')
    edges_num, edges_matrix= edges_dist(len(data_helper.vocab), data_helper, 1)
    model = torch.load(os.path.join('temp_model.pkl'))
    content, label = data_helper.get_content()

    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()
    graph_ed = []
    for core_word in core_words:#
        core_index = data_helper.vocab.index(core_word)
      
        results = {}
        unq_res = {}
        for i in range(len(data_helper.vocab)):
          
            word = data_helper.vocab[i]
            n_word = edges_matrix[i, core_index]
            
            if n_word != 0:
                  results[word] = edges_weights[n_word][0]
            else:
              continue
        
        for value, key in results.items():
          if value not in unq_res:
             unq_res[value] = key


        sort_results = sorted(unq_res.items(), key=lambda d: d[1])
        graph_ed.append(sort_results)
    # print(sort_results)
    return  graph_ed

def grab_core(core_words):
    cores = []
    cores_w = []
    # print('load model from file.')
    data_helper = DataHelper('test')
    edges_num, edges_matrix= edges_mapping1(len(data_helper.vocab), data_helper, 1)
    model = torch.load(os.path.join('temp_model.pkl'))
    content, label = data_helper.get_content()

    # eval_test(model,data_helper)

    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()
    graph_ed = []
    for core_word in core_words:#
        core_index = data_helper.vocab.index(core_word)
      
        results = {}
        for i in range(len(data_helper.vocab)):
          
            word = data_helper.vocab[i]
            n_word = edges_matrix[i, core_index]
            # n_word = edges_matrix[i, i]
            if n_word != 0:
                  cores.append(i)
                  cores_w.append(word)
                  
            else:
              continue
    return cores_w


## set key words to build a pre-defined graph
def predefine_core_(core_words, cores):
  core = []
  core_w = []
  max_ind = len(core_words)-1 

  for c in cores:
    indd = 0
    while indd <= max_ind:
        word = core_words[indd]
        
        if word == c:
          
          core_w.append(word)

        indd +=1
  return core_w

core_wds = []
key_core = ['i', 'my', 'me', 'myself', 'mine','depressed', 'interest','sleep','move']

def graph_eval_extend(core_words):
    
    # print('load model from file.')
    data_helper = DataHelper('test')
    edges_num, edges_matrix= edges_mapping1(len(data_helper.vocab), data_helper, 1)
    model = torch.load(os.path.join('temp_model.pkl'))
    content, label = data_helper.get_content()
    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()
    graph_ed = []
    
    for core_word in core_words:#
        other = []
        core_index = data_helper.vocab.index(core_word)
        for cc in core_words:
            if cc != core_word:
              o_ind = data_helper.vocab.index(cc)
              other.append(o_ind)

        results = {}
        for i in range(len(other)):
            word = data_helper.vocab[other[i]]
            n_word = edges_matrix[other[i], core_index]
            results[word] = edges_weights[n_word][0]
        unq_res = {}
        for value, key in results.items():
          if value not in unq_res:
             unq_res[value] = key

        sort_results = sorted(unq_res.items(), key=lambda d: d[1])
        graph_ed.append(sort_results)
    # print(sort_results)
    return  graph_ed

## personal defined:
def predefine_core(core_words):
  core = []
  
  cores = ['i', 'my', 'me', 'myself', 'mine','depressed', 'sleep','interest','move']
  max_ind = len(core_words)-1 

  for c in cores:
    indd = 0
    while indd <= max_ind:
        word = core_words[indd]
        indd +=1
        if word == c:
          core.append(c)
 
  return core

pred_f = predefine_core(core_words)
print(pred_f)

## defined the threshold of edge weights
threshold1 = 0.99
threshold2 = 0.99

def generate_graph(core_words):
    G = nx.Graph()

    index =0
    max_index = len(core_words)
    graph_edges = graph_eval(core_words)
    for node_list in graph_edges:
      if index < max_index:
        for nn in node_list:
          if nn[1] >= threshold1:
            G.add_edge(nn[0],core_words[index], weight=nn[1])
      index+=1
    return G

### new version
def generate_graph_ex(core_words):
    G = nx.Graph()
    index =0
    max_index = len(core_words)
    graph_edges = graph_eval_extend(core_words)
    for node_list in graph_edges:
      if index < max_index:
        for nn in node_list:
          if nn[1] >= threshold2:
            G.add_edge(nn[0],core_words[index], weight=nn[1])
      index+=1
    return G


def preprocess_graph(G):
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        if weight >=1:
            edge[2]['color'] = 'r'
        else:
            edge[2]['color'] = '#d260ff'


###  visualize results generated by GNN model######
G = generate_graph(list(set(core_words)))
print("graph has %d nodes with %d edges"%(nx.number_of_nodes(G),nx.number_of_edges(G)))
print(nx.number_connected_components(G),"connected components")

try:
 import pydotplus
 from networkx.drawing.nx_pydot import graphviz_layout
except ImportError:
 raise ImportError("This example needs Graphviz and either "
                       "PyGraphviz or PyDotPlus")

plt.figure(figsize=(40,50))
poss=graphviz_layout(G, prog="twopi")
# poss = nx.spring_layout(G)
nx.draw_networkx_nodes(G,poss,nodelist=G.nodes(),node_size=30,\
linewidths=0.1,vmin=0,vmax=1,alpha=0.8, with_labels=False)
nx.draw_networkx_edges(G,poss,edgelist=G.edges(),width=0.1,edge_color="black",alpha=0.6)
nx.draw_networkx_labels(G, poss, font_size=24, font_family='sans-serif')
# adjust the plot limits
xmax=1.02*max(xx for xx,yy in poss.values())
ymax=1.02*max(yy for xx,yy in poss.values())
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.axis('off')
plt.tight_layout()
plt.show()


