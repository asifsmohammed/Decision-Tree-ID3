# decision_tree.py

#Submitted By:
#Asif Sohail Mohammed - AXM190041
#Pragya Nagpal - PXN190012

#LINE 15: For using graphviz path of where graphviz is installed in anaconda is added to environment variable 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Pragya/Anaconda3/Library/bin/graphviz/'

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
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    z = partition(y)
    e = 0
    for key in z:
        e += -z.get(key)/len(y) * np.log2(z.get(key)/len(y))
    return e
    raise Exception('Function not yet implemented!')


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
    e_temp = len(x) / len(y) * entropy(y[x])
    return e - e_temp


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
    elif(max_depth == depth):
        maximum = max(z.values())
        for key, value in z.items():
            if value == maximum:
                return key

    else:
        max_ig = 0
        for i in range(len(x[0])):
            values = np.unique(x[:,i])
            for j in values:
                ig = mutual_information(np.where(x[:, i] == j)[0],y)
                if(ig > max_ig):
                    max_ig = ig
                    max_ig_tup = (i, j)
        attribute_value_pairs.remove(max_ig_tup)
        
        # Recursive call
        return {(max_ig_tup[0], max_ig_tup[1], True) : id3(x[np.where(x[:, max_ig_tup[0]] == max_ig_tup[1])], y[np.where(x[:, max_ig_tup[0]] == max_ig_tup[1])], 
                                attribute_value_pairs, depth+1, max_depth),
        (max_ig_tup[0], max_ig_tup[1], False) : id3(x[np.where(x[:, max_ig_tup[0]] != max_ig_tup[1])], y[np.where(x[:, max_ig_tup[0]] != max_ig_tup[1])], 
                    attribute_value_pairs, depth+1, max_depth)}
        
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

def conf_matrix(ytst, y_pred):
    # Confusion Matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(ytst)):
        if(ytst[i] == 1 and y_pred[i] == 1):
            tp += 1
        elif(ytst[i] == 0 and y_pred[i] == 0):
            tn += 1
        elif(ytst[i] == 0 and y_pred[i] == 1):
            fp += 1
        else:
            fn += 1
    cm = np.zeros( (2, 2), dtype=np.int32 )
    cm[0][0] = tp
    cm[0][1] = fn
    cm[1][0] = fp
    cm[1][1] = tn
    return cm

def defaultDT(Xtrn,ytrn,Xtst,ytst,name):
    classify = tree.DecisionTreeClassifier()
    classify = classify.fit(Xtrn, ytrn)
    ypred=classify.predict(Xtst)

    print("Accuracy:", metrics.accuracy_score(ytst, ypred))

    dot_data = tree.export_graphviz(classify, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render(name) #for saving the decision tree as pdf with name
    print(conf_matrix(ytst, ypred))

if __name__ == '__main__':
    # MONK's 1
    # Load the training data
    M = np.genfromtxt('./data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    #default sklearn model for Monks-1 (part C of assignment)
    print("Sklearn default decision Tree for monks-1")
    defaultDT(Xtrn,ytrn,Xtst,ytst,"monks-1")
    
    x = np.arange(1, 11)
    test_error1 = []
    train_error1 = []
    for i in range(1, 11):    #calculation on monks-1 for depth=1..10 (part A of assignment)
        print("depth is", i)
        # Learn a decision tree of depth 3
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        visualize(decision_tree)

        # Compute the train error
        y_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred)

        print('Monk\'s 1 Train Error = {0:4.2f}%.'.format(trn_err * 100))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)

        # Confusion Matrix for monks 1 depth 1 and 2 (part B of assignment)
        if(i == 1):
            monks1_depth1_cm = conf_matrix(ytst, y_pred)
        elif(i == 2):
            monks1_depth2_cm = conf_matrix(ytst, y_pred)

        print('Monk\'s 1 Test Error = {0:4.2f}%.'.format(tst_err * 100))
        
        train_error1.append(trn_err * 100)
        test_error1.append(tst_err * 100)
        
    fig,axes = plt.subplots()
    axes.plot(x, train_error1, label='Train Error')
    axes.plot(x, test_error1, label='Test Error')
    axes.set_xticks(x)
    axes.set_xlabel("Depth")
    axes.set_ylabel("Error")
    axes.set_title("MONK's 1 Train vs Test Error")
    axes.legend()
    plt.show()

    # MONK's 2
    # Load the training data
    M = np.genfromtxt('./data/monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    test_error2 = []
    train_error2 = []
    for i in range(1, 11):    #calculation on monks-2 for depth=1..10 (part A of assignment)
        print("Depth is", i)
        # Learn a decision tree of depth 3
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        visualize(decision_tree)

        # Compute the train error
        y_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred)

        print('Monk\'s 2 Train Error = {0:4.2f}%.'.format(trn_err * 100))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)

        print('Monk\'s 2 Test Error = {0:4.2f}%.'.format(tst_err * 100))
        
        train_error2.append(trn_err * 100)
        test_error2.append(tst_err * 100)
        
    fig,axes = plt.subplots()
    axes.plot(x, train_error2, label='Train Error')
    axes.plot(x, test_error2, label='Test Error')
    axes.set_xticks(x)
    axes.set_xlabel("Depth")
    axes.set_ylabel("Error")
    axes.set_title("MONK's 2 Train vs Test Error")
    axes.legend()
    plt.show()

    # MONK's 3
    # Load the training data
    M = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    test_error3 = []
    train_error3 = []
    for i in range(1, 11):   #calculation on monks-3 for depth=1..10 (part A of assignment)
        # Learn a decision tree of depth 1 to 10
        print("Depth is", i)
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        visualize(decision_tree)

        # Compute the train error
        y_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred)

        print('Monk\'s 3 Train Error = {0:4.2f}%.'.format(trn_err * 100))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)

        print('Monk\'s 3 Test Error = {0:4.2f}%.'.format(tst_err * 100))

        train_error3.append(trn_err * 100)
        test_error3.append(tst_err * 100)
        
    fig,axes = plt.subplots()
    axes.plot(x, train_error3, label='Train Error')
    axes.plot(x, test_error3, label='Test Error')
    axes.set_xticks(x)
    axes.set_xlabel("Depth")
    axes.set_ylabel("Error")
    axes.set_title("MONK's 3 Train vs Test Error")
    axes.legend()
    plt.show()

    

    # Part D Other Data Sets-Balance-scale
    # Load the training data
    M = np.genfromtxt('./data/balance-scale.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/balance-scale.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    #default sklearn model for Balance Scale
    print("Sklearn default decision Tree for balance-scale")
    defaultDT(Xtrn,ytrn,Xtst,ytst,"balance-scale")
    
    test_error_other = []
    train_error_other = []
    for i in range(1, 11):
        # Learn a decision tree of depth 1 to 10
        print("Depth is", i)
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        visualize(decision_tree)

        # Compute the train error
        y_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred)

        print('Monk\'s 3 Train Error = {0:4.2f}%.'.format(trn_err * 100))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)

        # Confusion Matrix
        if(i == 1):
            other_depth1_cm = conf_matrix(ytst, y_pred)
        elif(i == 2):
            other_depth2_cm = conf_matrix(ytst, y_pred)

        print('Monk\'s 3 Test Error = {0:4.2f}%.'.format(tst_err * 100))

        train_error_other.append(trn_err * 100)
        test_error_other.append(tst_err * 100)
        
    fig,axes = plt.subplots()
    axes.plot(x, train_error_other, label='Train Error')
    axes.plot(x, test_error_other, label='Test Error')
    axes.set_xticks(x)
    axes.set_xlabel("Depth")
    axes.set_ylabel("Error")
    axes.set_title("Breast Cancer Train vs Test Error")
    axes.legend()
    plt.show()

    
    print("Confusion matrix for Monk's 1 Depth 1 is\n", monks1_depth2_cm)
    print("Confusion matrix for Monk's 1 Depth 2 is\n", monks1_depth2_cm)
    print("Confusion matrix for Balance-Scale Dataset Depth 1 is\n", other_depth1_cm)
    print("Confusion matrix for Balance-Scale DataSet Depth 2 is\n", other_depth2_cm)
