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


## all words in a doc
core_words = []
data_helper = DataHelper('test')
for i in range(len(data_helper.vocab)):
    word = data_helper.vocab[i]
    core_words.append(word)

# print(core_words)

def graph_eval(core_words):
    
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
              # if edges_weights[n_word][0]>=0.95:
                  # print(n_word)
                  cores.append(i)
                  cores_w.append(word)
                  
            else:
              continue
    return cores_w



## personal defined:
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

    # eval_test(model,data_helper)

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
threshold1 = 0.98
threshold2 = 0.95

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
        # # False positives.
        # elif weight <1 and weight >=0.8:
        #     edge[2]['color'] = 'k'
        # # False negatives.
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


