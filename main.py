import numpy as np
import pandas as pd
import random
# from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.stats.contingency_tables import mcnemar

def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Fit tree to the training data with specified hyperparameters. Return tree object 

    x: attribute values
    y: class labels
    nmin: minimum number of observations in node for split
    minleaf: minimum number of observations required for a leaf node
    nfeat: number of features considered for each split 
    
    returns grown tree 
    """

    #stop if pure node or fewer than nmin samples, creates leaf node
    if len(y) < nmin or len(set(y)) == 1:  
        return TreeNode(label=majority_class(y))

    features_to_consider = np.random.choice(range(x.shape[1]), nfeat, replace=False) #for random forests, bagging

    best_feature, best_split_point, best_impurity = None, None, 1 #initialization, 1 is more than the max gini could be

    for feature_index in features_to_consider: # to find the best splitting point across all features
        split_point, impurity_value = best_split(x[:, feature_index], y, minleaf) #best splitting point for each feature
        if split_point is None:
            continue

        if impurity_value < best_impurity: 
            best_impurity = impurity_value
            best_feature = feature_index
            best_split_point = split_point

    if best_feature is None or best_split_point is None:
        return TreeNode(label=majority_class(y))    # if no splitting point is found, then we have a leaf node
    
    #now we will split the node's data to the two children
    left_mask = x[:, best_feature] <= best_split_point
    right_mask =  x[:, best_feature] > best_split_point

    #recursively grow a tree for each child, which will be in reality a branch of the initial tree
    left_child = tree_grow(x[left_mask], y[left_mask], nmin, minleaf, nfeat)
    right_child = tree_grow(x[right_mask], y[right_mask], nmin, minleaf, nfeat)
    tree = TreeNode(feature=best_feature, split_thr=best_split_point, left_child=left_child, right_child=right_child)

    return tree
            
def tree_pred(x, tr):
    """
    x: 2D numpy array (n_samples, n_features) 
    tr: The decision tree object (root node) created in tree_grow

    returns array of predictions for given tree
    """
    return np.array([predict_single(row, tr) for row in x])


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    Draws m bootstrap samples and returns a list of m trees. Extension of tree_grow 

    x: attribute values
    y: class labels
    nmin: minimum number of observations in node for split
    minleaf: minimum number of observations required for a leaf node
    nfeat: number of features considered for each split 
    m: number of bootstrap samples to be drawn

    returns a list of trees
    """    

    # initialize list  
    ensemble = []
    
    for i in range(m):
        # draw bootstrap sample
        x_sampled, y_sampled = draw_sample(x, y)
        # fit a tree to drawn sample and append to list 
        ensemble.append(tree_grow(x_sampled, y_sampled, nmin, minleaf, nfeat))

    return ensemble


def tree_pred_b(x, trees):
    """
    Gets predictions from m trees for all data points

    x: 2D numpy array (n_samples, n_features) 
    tr: The decision tree object (root node) created in tree_grow

    returns an array with the model's predictions
    """

    # get model's predictions 
    votes = [tree_pred(x, tr) for tr in trees]

    # Transpose so that every row has the votes for a sample 
    votes_array =  np.vstack(votes).T
    
    preds = [np.argmax(np.bincount(votes_array[i])) for i in range(votes_array.shape[0])]

    return np.array(preds)

class TreeNode:
    def __init__(self, feature=None, split_thr=None, left_child=None, right_child=None, label=None, is_root = False):
        self.feature = feature       # index for the feature for which the node splits
        self.split_thr = split_thr   # splitting point value for the above feature
        self.left_child = left_child 
        self.right_child = right_child # these (left/right child) point to the children    
        self.label = label           # if it is leaf, it will show what class/label it predicts
   
    def is_leaf(self):
        return self.label is not None  # boolean, if it is leaf or not (so that we know if it is the classifier node)


def gini_index(y_array):
    """
    Calculates gini impurity for given split from the labels of the samples in the node
    
    returns calculated gini value
    """
    
    # count 0s and 1s in label column
    num_zeros = (y_array == 0).sum()
    num_ones = (y_array == 1).sum()
    # num of samples in node
    total = num_zeros + num_ones
    if total == 0:  
        return 0
    # calculate class probs 
    p_0 = num_zeros / total
    p_1 = num_ones / total
    
    return 1 - (p_0 ** 2 + p_1 ** 2)


# Returns splits: dictionary of the possible splitting point and the calculated impurity for them 
def define_splits(x, y, minleaf):
    sorted_col = np.sort(np.unique(x))
    splitpoints = (sorted_col[0:-1] + sorted_col[1:])/2

    splits = {}
    for c in splitpoints:
        left_ch = y[x<=c]
        right_ch = y[x>c]

        #If minleaf constraint is violated, rather than possibly choosing this split
        #resulting in an immediate leaf, setting the value to 1 allows for the program
        #to still look for other splits
        if left_ch.size < minleaf or right_ch.size < minleaf:
            splits[c] = 1
            continue
        
        splits[c] = (len(left_ch)/len(y))*gini_index(left_ch) + (len(right_ch)/len(y))*gini_index(right_ch)

    return splits

def best_split(x, y, minleaf):
    """
    Recursive function to calculate the best split for a single column x

    x: array/feature column for which to find best split point
    y:  array/ label column 
    minleaf: minimum number of observations required for a leaf node

    returns split index and dictionary
    """
    split_dict = define_splits(x, y, minleaf)
    
    if not split_dict:
        return None, None # if no splitting points are found for the particular feature, pass 
    split_value = min(split_dict, key=split_dict.get) #the key in the dictionary (splitting point) for which the value (impurity) is minimized
    
    return split_value, split_dict[split_value]

def majority_class(y):
    # function to find majority class from label column
    num_zeros = (y == 0).sum()
    num_ones = (y == 1).sum()
    if num_ones>=num_zeros:
        return 1
    else:
        return 0
    # return np.argmax(np.bincount(y)) # this is used for leaf node's utility of predicting the majority class


def predict_single(instance, node):
    """
    Predicts class for a single example. Used in tree_pred function.

    instance: single example from the data
    node: tree
    returns model prediction for the single sample
    """
    while not node.is_leaf(): #go through the nodes until we reach the leaf one that is the final classifier
        if instance[node.feature] <= node.split_thr:
            node = node.left_child 
        else:
            node = node.right_child 
    return node.label 
    

def draw_sample(x, y):
    """
    Performs sampling with replacement on the training data.

    x: attribute values
    y: class labels

    returns arrays with sample data and their labels
    """
    n_samples = len(y)
    # choose n_samples randomly with replacement
    sample_idx = random.choices(range(n_samples), k=n_samples)
    
    x_sampled = np.array([x[i] for i in sample_idx])
    y_sampled = np.array([y[i] for i in sample_idx])

    return x_sampled, y_sampled


# def write_results(filename, train_preds, y_train, test_preds, y_test):
#     """
#     Writes results for all models on train and test splits to a text file and saves confusion matrices as png files
    
#     filename: how the text file is going to be named
#     train_preds: predictions on train data
#     y_train: train set labels
#     test_preds: predictions on test data
#     y_test: test set labels
#     """
#     with open(f'./{filename}.txt', 'w') as f:
#         f.write('Train Results')
#         f.write(f'\n accuracy: {accuracy_score(train_preds, y_train)}') 
#         f.write(f'\n precision: {precision_score(train_preds, y_train)}') 
#         f.write(f'\n recall: {recall_score(train_preds, y_train)}') 

#         f.write('\nTest Results')
#         f.write(f'\n accuracy: {accuracy_score(test_preds, y_test)}') 
#         f.write(f'\n precision: {precision_score(test_preds, y_test)}') 
#         f.write(f'\n recall: {recall_score(test_preds, y_test)}') 

#     # plot and save confusion matrix
#     train_cm = confusion_matrix(y_train, train_preds)
#     sns.heatmap(train_cm, annot=True, fmt="d")
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.savefig(fname=f'cm_train_{filename}')
#     plt.close()
    
#     test_cm = confusion_matrix(y_test, test_preds)
#     sns.heatmap(test_cm, annot=True, fmt="d")
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.savefig(fname=f'cm_test_{filename}')
#     plt.close()

#     return


# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(suppress=True)


# # load train and test data from csv files
# data = np.genfromtxt('./eclipse_train.csv', delimiter=',', skip_header=1)
# test_data = np.genfromtxt('./eclipse_test.csv', delimiter=',', skip_header=1)

# # train data
# x_train = data[:, :-1]
# y_train = data[:,-1]
# # test data
# x_test = test_data[:, :-1]
# y_test = test_data[:,-1]

# # Single tree
# tr = tree_grow(x_train, y_train, 15, 5, 41)
# preds_train = tree_pred(x_train, tr)
# preds_test = tree_pred(x_test, tr)
# write_results('single_tree', preds_train, y_train, preds_test, y_test)
# # Bagging
# trees = tree_grow_b(x_train, y_train, 15, 5, 41, 100)
# preds_train_bag = tree_pred_b(x_train, trees)
# preds_test_bag = tree_pred_b(x_test, trees)
# write_results('bagging', preds_train_bag, y_train, preds_test_bag, y_test)
# # Random Forest
# forest = tree_grow_b(x_train, y_train, 15, 5, 6, 100)
# preds_train_for = tree_pred_b(x_train, forest)
# preds_test_for = tree_pred_b(x_test, forest)
# write_results('random_forest', preds_train_for, y_train, preds_test_for, y_test)



# def stat_test_mcnemar(pred_1, pred_2, test):

#     #each vector has 1 in correct predictions and 0 in wrong ones 
#     correct_1 = np.where(pred_1 == test, 1, 0)  
#     correct_2 = np.where(pred_2 == test, 1, 0) 
#     #calculate cont table terms and define cont table
#     a = np.sum((correct_1 == 1) & (correct_2 == 1))  
#     b = np.sum((correct_1 == 1) & (correct_2 == 0))  
#     c = np.sum((correct_1 == 0) & (correct_2 == 1))  
#     d = np.sum((correct_1 == 0) & (correct_2 == 0))   
#     contingency_table = np.array([[a, b], [c, d]])
#     result = mcnemar(contingency_table, exact=False, correction=True)

#     return result


# result_A = stat_test_mcnemar(preds_test, preds_test_bag, y_test)
# result_B = stat_test_mcnemar(preds_test, preds_test_for, y_test)
# result_C = stat_test_mcnemar(preds_test_for, preds_test_bag, y_test)

# print("tree+bag: \n", result_A)
# print("tree+for: \n", result_B)
# print("for+bag: \n", result_C)
