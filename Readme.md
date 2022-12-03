# GraphSAGE-pytorch-inductive

## 动机 Motivation

论文Inductive Representation Learning on Large Graphs的tensorflow版本实现是inductive的，但在graphsage-pytorch中(https://github.com/twjiang/graphSAGE-pytorch)确实transductive的。

The tensorflow version of implementation of graphsage is inductive, however the pytorch version (https://github.com/twjiang/graphSAGE-pytorch) is transductive.

这是因为作者选用的训练集的邻居中包含了测试集和验证集的节点，也就是说，部分测试集中的节点在训练过程中是可知的。

This is because the author included neighors of test set and val dataset in train set. And this means part of the nodes in test set is known in the training period.

所以，我对此进行了改进，将graphsage的代码从transductive转变为inductive。

So, I made some changes to change it from transductive to inductive.

## 方法 Methods

我采用的数据集是cora，在建立adj_lists的时候，对于验证集和测试集中的节点，直接建立边即可，而对于训练集，则需要判断两个节点是否都在训练集中，只有同时在训练集中才建立双向的边，这样就保证了在训练时只运用了训练集中的数据，从而确保是inductive方法。

I use cora as the dataset. When building adj_lists, for nodes in test and val set, direct adding an edge to them is OK; for train set, we need to judge whether both two nodes are in train set, we can add edges to them only when the answer is true to ensure we use no nodes in val or test set when do the training.

```python
# 对于测试集和验证集，直接建立adj_list
if node_map[pair[0]] not in train_index: 
adj_lists[node_map[pair[0]]].add(node_map[pair[1]])
if node_map[pair[1]] not in train_index: 
adj_lists[node_map[pair[1]]].add(node_map[pair[0]])

# 对于训练集，neighbor只能包含训练集中的节点
if node_map[pair[1]] in train_index and node_map[pair[1]] in train_index: 
adj_lists[node_map[pair[0]]].add(node_map[pair[1]])
adj_lists[node_map[pair[1]]].add(node_map[pair[0]])
```

### 对训练集中的孤立节点的处理： Dealing with isolated nodes:

由于我们将训练集中的节点邻居设置为训练集中的节点，那么就可能出现如下问题：某个训练集中的节点的所有邻居都在测试集或验证集中，导致在训练过程中，他实际上成了没有邻居的孤立节点。

孤立节点带来的问题是，当计算损失时，需要对节点进行正采样和负采样，我们无法对孤立节点进行正采样。

Using the methods above, the following problems may occur: All the neighbors of a train set's node are in test set or val set, then it turns an isolated node without neighbors.

The problem with isolated nodes is that when calculating losses, we need to do positive and negative sampling of the nodes, and we cannot do positive sampling of isolated nodes.

#### 方法1：增加自循环 Method1: add self-loop

在为节点建立adj_list的时候，为节点加上自循环。

When building adj_list, add self-loop for nodes.


```python
if node_map[pair[0]] not in adj_lists: adj_lists[node_map[pair[0]]].add(node_map[pair[0]])
if node_map[pair[1]] not in adj_lists: adj_lists[node_map[pair[1]]].add(node_map[pair[1]])
```

#### 方法2：直接删除train中孤立节点

赋初值为空

First set the initial value as null.

```python
if node_map[pair[0]] not in adj_lists: adj_lists[node_map[pair[0]]] = set()
if node_map[pair[1]] not in adj_lists: adj_lists[node_map[pair[1]]] = set()
```

删除孤立节点的adj_list和train_index

Then delete the adj_list and train_index of isolated nodes

```python
i = 0
while i < len(train_index):
		if i<len(train_index) and len(adj_lists[train_index[i]]) == 0:
		    # print('--------------deleted--------------', i)
		    del adj_lists[train_index[i]]
		    train_index = np.delete(train_index, i)
		    i -= 1
    i += 1
```

## 代码介绍

myGraphSAGE_inductive_delete.py : 采用删除孤立节点的方式的graphsage的inductive版本
myGraphSAGE_inductive_selfloop.py : 采用增加自循环的方式的graphsage的inductive版本
myGraphSAGE_transductive.py : 原始的graphsage的transductive版本

myGraphSAGE_inductive_delete.py : The inductive version of graphsage by deleting isolated nodes
myGraphSAGE_inductive_selfloop.py : The inductive version of graphsage by adding self-loop
myGraphSAGE_transductive.py : the raw transductive version of graphsage

运行方式： How to run:
```
python3 + xxx.py
```