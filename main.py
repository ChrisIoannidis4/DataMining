import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature=None, split_thr=None, left_child=None, right_child=None, label=None, is_root = False):
        self.feature = feature       # index for the feature for which the node splits
        self.split_thr = split_thr   # splitting point value for the above feature
        self.left_child = left_child 
        self.right_child = right_child # these (left/right child) point to the children    
        self.label = label           # if it is leaf, it will show what class/label it predicts
        # self.is_root = is_root    
    def is_leaf(self):
        return self.label is not None  # boolean, if it is leaf or not (so that we know if it is the classifier node)

class DecisionTree:
    def __init__(self, rootNode=None):
        self.rootNode=rootNode   #the decision tree's root node 


# def impurity(y_array):
#     num_zeros = (y_array == 0).sum()
#     num_ones = (y_array == 1).sum()
#     return (num_zeros*num_ones)/(y_array.size **2)

def gini_index(y_array):
    num_zeros = (y_array == 0).sum()
    num_ones = (y_array == 1).sum()
    total = num_zeros + num_ones
    if total == 0:  
        return 0
    p_0 = num_zeros / total
    p_1 = num_ones / total
    return 1 - (p_0 ** 2 + p_1 ** 2)

def define_splits(x, y): 
    #returns splits: dictionary of the possible splitting point and the calculated impurity for them 
    sorted_col = np.sort(np.unique(x))
    splitpoints = (sorted_col[0:-1] + sorted_col[1:])/2 
    splits = {}
    for i, c in enumerate(splitpoints):
        left_ch = y[x>c]
        right_ch = y[x<=c]

        splits[c] = (len(left_ch)/len(y))*gini_index(left_ch) + (len(left_ch)/len(y))*gini_index(right_ch)
    return splits

def best_split(x,y):
    split_dict = define_splits(x,y)
    if not split_dict:
        return None, None # if no splitting points points are found for the particular feature, pass 
    split_value = min(split_dict, key=split_dict.get) #the key in the dictionary (splitting point) for which the value (impurity) is minimized
    return split_value, split_dict[split_value]


def majority_class(y):
    num_zeros = (y == 0).sum()
    num_ones = (y == 1).sum()
    if num_ones>=num_zeros:
        return 1
    else:
        return 0
    # return np.argmax(np.bincount(y)) # this is used for leaf node's utility of predicting the majority class

def tree_grow(x, y, nmin, minleaf, nfeat):

    if len(y) < nmin or len(set(y)) == 1:  #stopping if pure node or fewer than nmin samples, creates leaf node
        return TreeNode(label=majority_class(y))

    features_to_consider = np.random.choice(x.shape[1], nfeat, replace=False) #for random forests, bagging

    best_feature, best_split_point, best_impurity = None, None, 1 #initialization, 1 is more than the max gini could be

    for feature_index in features_to_consider: # to find the best splitting point across all features
        split_point, impurity_value = best_split(x[:, feature_index], y) #best splitting point for each feature

        if impurity_value < best_impurity: 
            best_impurity = impurity_value
            best_feature = feature_index
            best_split_point = split_point

    if best_feature is None or best_split_point is None:
        return TreeNode(label=majority_class(y))    # if no splitting point is found, then we have a leaf node
    
    #now we will split the node's data to the two children
    left_mask = x[:, best_feature] > best_split_point
    right_mask =  x[:, best_feature] < best_split_point

    #recursively grow a tree for each child, which will be in reality a branch of the initial tree
    left_child = tree_grow(x[left_mask], y[left_mask], nmin, minleaf, nfeat)
    right_child = tree_grow(x[right_mask], y[right_mask], nmin, minleaf, nfeat)
    node = TreeNode(feature=best_feature, split_thr=best_split_point, left_child=left_child, right_child=right_child)

    return node, DecisionTree(rootNode=node)
            

def predict_single(instance, node):
    while not node.is_leaf(): #go through the nodes until we reach the leaf one that is the final classifier
        if instance[node.feature] > node.split_thr:
            node = node.left_child 
        else:
            node = node.right_child 
    return node.label 


def tree_pred(x, tr):
    """
    x: 2D numpy array (n_samples, n_features) 
    tr: The decision tree object (root node) created in tree_grow
    y: 1D numpy array of predicted class labels
    """
    return np.array([predict_single(row, tr) for row in x])
    


data = pd.read_csv('./change.csv', sep=',', header= None)
print(data.head())

_, tr = tree_grow(data.to_numpy()[:,:-1], data.to_numpy()[:,-1], 5, 1, 7)

# y =  data.to_numpy()[:,-1]
# np.argmax(np.bincount(y))