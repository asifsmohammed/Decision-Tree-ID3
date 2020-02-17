 # M = np.genfromtxt('c:/Users\moham\Decision-Tree-ID3\monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytrn = M[:, 0]
    # Xtrn = M[:, 1:]

    # # Load the test data
    # M = np.genfromtxt('c:/Users\moham\Decision-Tree-ID3\monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytst = M[:, 0]
    # Xtst = M[:, 1:]

    # test_error2 = []
    # train_error2 = []
    # for i in range(1, 11):
    #     print("Depth is", i)
    #     # Learn a decision tree of depth 3
    #     decision_tree = id3(Xtrn, ytrn, max_depth=i)
    #     visualize(decision_tree)

    #     # Compute the train error
    #     y_pred = [predict_example(x, decision_tree) for x in Xtrn]
    #     trn_err = compute_error(ytrn, y_pred)

    #     print('Monk\'s 2 Train Error = {0:4.2f}%.'.format(trn_err * 100))

    #     # Compute the test error
    #     y_pred = [predict_example(x, decision_tree) for x in Xtst]
    #     tst_err = compute_error(ytst, y_pred)

    #     print('Monk\'s 2 Test Error = {0:4.2f}%.'.format(tst_err * 100))
        
    #     train_error2.append(trn_err * 100)
    #     test_error2.append(tst_err * 100)
        
    # fig,axes = plt.subplots()
    # axes.plot(x, train_error2, label='Train Error')
    # axes.plot(x, test_error2, label='Test Error')
    # axes.set_xticks(x)
    # axes.set_xlabel("Depth")
    # axes.set_ylabel("Error")
    # axes.set_title("MONK's 2 Train vs Test Error")
    # axes.legend()
    # plt.show()

    # # MONK's 3
    # # Load the training data
    # M = np.genfromtxt('c:/Users\moham\Decision-Tree-ID3\monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytrn = M[:, 0]
    # Xtrn = M[:, 1:]

    # # Load the test data
    # M = np.genfromtxt('c:/Users\moham\Decision-Tree-ID3\monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytst = M[:, 0]
    # Xtst = M[:, 1:]

    # test_error3 = []
    # train_error3 = []
    # for i in range(1, 11):
    #     # Learn a decision tree of depth 1 to 10
    #     print("Depth is", i)
    #     decision_tree = id3(Xtrn, ytrn, max_depth=i)
    #     visualize(decision_tree)

    #     # Compute the train error
    #     y_pred = [predict_example(x, decision_tree) for x in Xtrn]
    #     trn_err = compute_error(ytrn, y_pred)

    #     print('Monk\'s 3 Train Error = {0:4.2f}%.'.format(trn_err * 100))

    #     # Compute the test error
    #     y_pred = [predict_example(x, decision_tree) for x in Xtst]
    #     tst_err = compute_error(ytst, y_pred)

    #     print('Monk\'s 3 Test Error = {0:4.2f}%.'.format(tst_err * 100))

    #     train_error3.append(trn_err * 100)
    #     test_error3.append(tst_err * 100)
        
    # fig,axes = plt.subplots()
    # axes.plot(x, train_error3, label='Train Error')
    # axes.plot(x, test_error3, label='Test Error')
    # axes.set_xticks(x)
    # axes.set_xlabel("Depth")
    # axes.set_ylabel("Error")
    # axes.set_title("MONK's 3 Train vs Test Error")
    # axes.legend()
    # plt.show()

    

    # # d. Other Data Sets
    # # Load the training data
    # M = np.genfromtxt('c:/Users\moham\Decision-Tree-ID3\\balance-scale.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytrn = M[:, 0]
    # Xtrn = M[:, 1:]

    # # Load the test data
    # M = np.genfromtxt('c:/Users\moham\Decision-Tree-ID3\\balance-scale.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    # ytst = M[:, 0]
    # Xtst = M[:, 1:]

    # test_error_other = []
    # train_error_other = []
    # for i in range(1, 11):
    #     # Learn a decision tree of depth 1 to 10
    #     print("Depth is", i)
    #     decision_tree = id3(Xtrn, ytrn, max_depth=i)
    #     visualize(decision_tree)

    #     # Compute the train error
    #     y_pred = [predict_example(x, decision_tree) for x in Xtrn]
    #     trn_err = compute_error(ytrn, y_pred)

    #     print('Monk\'s 3 Train Error = {0:4.2f}%.'.format(trn_err * 100))

    #     # Compute the test error
    #     y_pred = [predict_example(x, decision_tree) for x in Xtst]
    #     tst_err = compute_error(ytst, y_pred)

    #     # Confusion Matrix
    #     if(i == 1):
    #         other_depth1_cm = conf_matrix(ytst, y_pred)
    #     elif(i == 2):
    #         other_depth2_cm = conf_matrix(ytst, y_pred)

    #     print('Monk\'s 3 Test Error = {0:4.2f}%.'.format(tst_err * 100))

    #     train_error_other.append(trn_err * 100)
    #     test_error_other.append(tst_err * 100)
        
    # fig,axes = plt.subplots()
    # axes.plot(x, train_error_other, label='Train Error')
    # axes.plot(x, test_error_other, label='Test Error')
    # axes.set_xticks(x)
    # axes.set_xlabel("Depth")
    # axes.set_ylabel("Error")
    # axes.set_title("Balance Scale Train vs Test Error")
    # axes.legend()
    # plt.show()

    
    # print("Confusion matrix for Monk's 1 Depth 1 is\n", monks1_depth2_cm)
    # print("Confusion matrix for Monk's 1 Depth 2 is\n", monks1_depth2_cm)
    # print("Confusion matrix for Balance Scale Data Set Depth 1 is\n", other_depth1_cm)
    # print("Confusion matrix for Balance Scale Data Set Depth 2 is\n", other_depth2_cm)
