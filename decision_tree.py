# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    values, counts = np.unique(x, return_counts=True)
    return dict(zip(values, counts))
    #raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    z = partition(y)
    # print(z)
    e = 0
    for key in z:
        e += -z.get(key)/len(y) * np.log2(z.get(key)/len(y))
    return e
    #raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    e = entropy(y)

    e_temp = 0
    # print(x)
    # print("--->", np.extract(x, y))
    e_temp = len(x) / len(y) * entropy(np.extract(x , y))

    return e - e_temp

    # info_gain = 0
    # for ig in range(len(x[0])):
    #     e_temp = 0
    #     values, counts = np.unique(x[:,ig], return_counts=True)
    #     for i in range(len(values)):
    #         e_temp = counts[i] / len(y) * entropy(np.extract(x[:, ig] == values[i], y))
    #         print("*****", e_temp, "*****")
    #         ig_temp = e - e_temp
    #         print("Information Gain of x",ig, values[i], "is", ig_temp)
    #         if(ig_temp > info_gain):
    #             info_gain = ig_temp
    #             feature = ig
    #             value = i
    # print("Max Information gain is",info_gain, "for colum x", feature, value)
    # return feature, value


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.
    
    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    attribute_value_pairs = []
    for i in range(len(x[0])):
        values = np.unique(x[:,i])
        for j in values:
            tup = (i, j)
            attribute_value_pairs.append(tup)
    # Condition 1
    z = partition(y)
    if(len(z) == 1):
        return list(z.keys())[0]
    
    # Condition 2
    elif(len(attribute_value_pairs) == 0):
        maximum = max(z.values())
        for key, value in z.items():
            if value == maximum:
                return key
                
    # Condition 3
    elif(max_depth - depth == 0):
        maximum = max(z.values())
        for key, value in z.items():
            if value == maximum:
                return key

    else:
        max_ig = 0
        for i in range(len(x[0])):
            values = np.unique(x[:,i])
            for j in values:
                ig = mutual_information(np.extract(x[:, i] == j, x[:, i]), y)
                # print("For Tuple (%d, %d) Information Gain is %f" % (i , j, ig))
                if(ig > max_ig):
                    max_ig = ig
                    max_ig_tup = (i, j)
        # print("Maximum Information Gain", max_ig, max_ig_tup)
        attribute_value_pairs.remove(max_ig_tup)
        print("-------------->", max_ig_tup)
        
        # Recursive call
        return {(max_ig_tup[0], max_ig_tup[1], True) : id3(x[np.where(x[:, max_ig_tup[0]] == max_ig_tup[1])], y[np.where(x[:, max_ig_tup[0]] == max_ig_tup[1])], 
                                attribute_value_pairs, depth+1, 7),
        (max_ig_tup[0], max_ig_tup[1], False) : id3(x[np.where(x[:, max_ig_tup[0]] != max_ig_tup[1])], y[np.where(x[:, max_ig_tup[0]] != max_ig_tup[1])], 
                    attribute_value_pairs, depth+1, 7)}
        
    # raise Exception('Function not yet implemented!')

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for key, value in tree.items():
        if((x[key[0]] == key[1]) == key[2]):
            if(isinstance(value, dict)):
                return(predict_example(x, value))
            else:
                return value
    # raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    count =  0
    for i in range(len(y_pred)):
        if(y_true[i] != y_pred[i]):
            count += 1
    return (1 / len(y_true)) * count
    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('C:/Users\moham\OneDrive\Desktop\Machine Learning S20\PA\data\monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('C:/Users\moham\OneDrive\Desktop\Machine Learning S20\PA\data\monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    print(decision_tree)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    print(y_pred)
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
