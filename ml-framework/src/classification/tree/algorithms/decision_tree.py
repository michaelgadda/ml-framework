class BTnode:
  def __init__(self, c_o_d=None, split=None, parent=None, l_child=None, r_child=None, impurity_val=None, subset_indices=None, pred_val = None, feature=None):
    self.c_o_d = c_o_d
    self.feature = feature
    self.split =  split
    self.pred_val = pred_val
    self.impurity_val = impurity_val
    self.subset_indices = subset_indices
    self.parent = parent
    self.l_child = l_child
    self.r_child = r_child

class myDecisionTree:
  def __init__(self, max_depth=20, max_leaf_nodes=20):
    self.max_depth = max_depth
    self.max_leaf_nodes = max_leaf_nodes
    self.trained_dt = BTnode()

  #This assumes all features have been converted to ints if they were previously categorical
  def continuous_or_discrete(self, X, threshold_perc = .05, threshold_num = None):
    if threshold_num == None:
      # Use threshold Perc
      unique_feat = np.unique(X)
      num_sam = X.shape[0]
      if len(unique_feat)/num_sam > threshold_perc:
        return "continuous"
      else:
        return "discrete"
    else:
      # Use threshold Num
      unique_feat = np.unique(X)
      num_sam = X.shape[0]
      if len(unique_feat) > threshold_num:
        return "continuous", len(unique_feat)
      else:
        return "discrete"



  def discrete_split(self, X, Y, measure):
    if measure == 'MSE':
      node_subset_len = X.shape[0]
      #print(c_o_d)
      best_split = None
      best_mse = None
      pred_val = None
      d_len = X.shape[0]
      for value in set(X):
        l_node = np.take(Y, np.where(X==value))[0]
        l_len = l_node.shape[0]
        l_pred = value
        #print(f'l_node: {l_node}, len: {l_len}')
        l_mse = np.sum((l_pred - l_node)**2)/l_len
        #print(l_pred - l_node, f"val difference pred: {l_pred}, node: {l_node}, l_mse: {l_mse} difference: {(l_pred - l_node)**2} sum: {np.sum((l_pred - l_node)**2)}")
        r_node = np.take(Y, np.where(X!=value))[0]
        r_len = r_node.shape[0]
        r_pred = value
        r_mse = np.sum((r_pred - r_node)**2)/r_len
        total_mse = (l_len/d_len)*l_mse + (r_len/d_len)*r_mse
        if best_mse == None:
          best_mse = total_mse
          pred_val = l_pred
          best_split = value
        elif best_mse > total_mse:
          best_mse = total_mse
          pred_val = l_pred
          best_split = value
      return best_mse, best_split
    if measure == 'Gini':
      node_subset_len = X.shape[0]
      d_len = X.shape[0]
      best_class_impurity_vals = None
      best_split = None
      #print(node_subset_len)
      for value in set(X):
        l_node = np.take(Y, np.where(X==value))[0]
        l_len = l_node.shape[1]
        ly_value_counts = np.unique(l_node, return_counts = True)
        r_node = np.take(Y, np.where(X!=value))[0]
        r_len = r_node.shape[1]
        ry_value_counts = np.unique(r_node, return_counts = True)
        r_curr_gin_impurity = 1
        l_curr_gin_impurity = 1
        temp_class_impurity_val = 0
        for item in ry_value_counts[1]:
          # its the number of each class we have within the split
          r_curr_gin_impurity -= ((item/r_len)**2)
          #temp_impurity_vals.append(([index]/node_subset_len)**2)
        for item in ly_value_counts[1]:
          # its the number of each class we have within the split
          #print(f"Item: {item} l_len: {l_len} node_subset_len: {node_subset_len} ")
          l_curr_gin_impurity -= ((item/l_len)**2)
        temp_class_impurity_val = (r_curr_gin_impurity*node_subset_len + l_curr_gin_impurity*node_subset_len)/d_len
        if best_class_impurity_vals == None:
          best_class_impurity_vals = temp_class_impurity_val
          best_split = value
        elif best_class_impurity_vals > temp_class_impurity_val:
          best_class_impurity_vals = temp_class_impurity_val
          best_split = value
      return best_class_impurity_vals, best_split

  def continuous_split(self, X, Y, measure):
    if measure == 'MSE':
      node_subset_len = X.shape[0]
      #print(c_o_d)
      best_split = None
      best_mse = None
      pred_val = None
      sorted_indices = X.argsort()
      X = X[sorted_indices]
      Y = Y[sorted_indices]
      d_len = X.shape[0]
      for index in range(node_subset_len-1):
        average_of_ixs = (X[index] + X[index+1])/2
        l_node = np.take(Y, np.where(X<=average_of_ixs))[0]
        l_len = l_node.shape[0]

        l_pred = np.average(l_node)
        #print(f'l_node: {l_node}, len: {l_len}')
        l_mse = np.sum((l_pred - l_node)**2)/l_len
        #print(l_pred - l_node, f"val difference pred: {l_pred}, node: {l_node}, l_mse: {l_mse} difference: {(l_pred - l_node)**2} sum: {np.sum((l_pred - l_node)**2)}")
        r_node = np.take(Y, np.where(X>average_of_ixs))[0]
        r_len = r_node.shape[0]
        r_pred = np.average(r_node)
        r_mse = np.sum((r_pred - r_node)**2)/r_len
        total_mse = (l_len/d_len)*l_mse + (r_len/d_len)*r_mse
        if best_mse == None:
          best_mse = total_mse
          pred_val = l_pred
          best_split = average_of_ixs
        elif best_mse > total_mse:
          best_mse = total_mse
          pred_val = l_pred
          best_split = average_of_ixs
        #print(best_mse, pred_val, best_split)
      return best_mse, best_split
    if measure == 'Gini':
      node_subset_len = X.shape[0]
      sorted_indices = X.argsort()
      #print(sorted_indices, X)
      X = X[sorted_indices]
      Y = Y[sorted_indices]
      d_len = X.shape[0]
      best_class_impurity_vals = None
      best_split = None
      #print(node_subset_len)
      for index in range(node_subset_len-1):
        average_of_ixs = (X[index] + X[index+1])/2
        l_node = np.take(Y, np.where(X<=average_of_ixs))
        l_len = l_node.shape[1]
        ly_value_counts = np.unique(l_node, return_counts = True)
        r_node = np.take(Y, np.where(X>average_of_ixs))
        r_len = r_node.shape[1]
        ry_value_counts = np.unique(r_node, return_counts = True)
        r_curr_gin_impurity = 1
        l_curr_gin_impurity = 1
        temp_class_impurity_val = 0
        for item in ry_value_counts[1]:
          # its the number of each class we have within the split
          r_curr_gin_impurity -= ((item/r_len)**2)
          #temp_impurity_vals.append(([index]/node_subset_len)**2)
        for item in ly_value_counts[1]:
          # its the number of each class we have within the split
          #print(f"Item: {item} l_len: {l_len} node_subset_len: {node_subset_len} ")
          l_curr_gin_impurity -= ((item/l_len)**2)
        temp_class_impurity_val = (r_curr_gin_impurity*node_subset_len + l_curr_gin_impurity*node_subset_len)/d_len
        if best_class_impurity_vals == None:
          best_class_impurity_vals = temp_class_impurity_val
          best_split = average_of_ixs
        elif best_class_impurity_vals > temp_class_impurity_val:
          best_class_impurity_vals = temp_class_impurity_val
          best_split = average_of_ixs
      return best_class_impurity_vals, best_split

  def fit(self, X, Y):
    BT = BTnode()
    y_c_o_d = self.continuous_or_discrete(Y)
    self.trained_dt = self.find_best_split(X,Y, 0, BT, y_c_o_d)


  def find_best_split(self, X, Y, depth, tree, y_c_o_d):
    print(depth, "<--- Depth")
    if depth < 5 and X.shape[0] > 2:
      depth +=1
    else:
      return tree
    # i believe my understanding of gini impurity and regression is incorrect in terms of when to use it. I believe that we use regression gini impurity AWLAYS based on the target class; what may change is how we split the data - as we may have continuous of discrete variables. But if our output is always classification then I am pretty sure we can always use gini impurity

    best_class_impurity_vals = None
    best_split = np.inf
    best_c_o_d = None
    best_feature = 0  #<--- setting as place holder to appease the program

    static_X = copy(X)
    for index, column in enumerate(X.T):
      #print(column, "THIS IS A COLUMN^>")
      c_o_d = self.continuous_or_discrete(X)
      if c_o_d == "continuous":
        if y_c_o_d == "continuous":
          temp_class_impurity_vals, temp_split = self.continuous_split(column, Y, "MSE")
          if best_class_impurity_vals == None:
              best_feature = index
              best_class_impurity_vals, best_split =  temp_class_impurity_vals, temp_split
              best_c_o_d = c_o_d
              #print(best_split)
          elif best_class_impurity_vals > temp_class_impurity_vals:
              best_feature = index
              best_class_impurity_vals, best_split =  copy(temp_class_impurity_vals), copy(temp_split)
              #print(best_class_impurity_vals, best_split)
              best_c_o_d = c_o_d
        else:
          temp_class_impurity_vals, temp_split = self.continuous_split(column, Y, "Gini")
          print(temp_class_impurity_vals)
          if best_class_impurity_vals == None:
              best_feature = index
              best_class_impurity_vals, best_split =  temp_class_impurity_vals, temp_split
              best_c_o_d = c_o_d
              #print(best_split)
          elif best_class_impurity_vals > temp_class_impurity_vals:
              best_feature = index
              best_class_impurity_vals, best_split =  copy(temp_class_impurity_vals), copy(temp_split)
              #print(best_class_impurity_vals, best_split)
              best_c_o_d = c_o_d
      elif c_o_d == "discrete":
        if y_c_o_d == "continuous":
          temp_class_impurity_vals, temp_split = self.discrete_split(column, Y, "MSE")
          if best_class_impurity_vals == None:
              best_feature = index
              best_class_impurity_vals, best_split =  temp_class_impurity_vals, temp_split
              best_c_o_d = c_o_d
              #print(best_split)
          elif best_class_impurity_vals > temp_class_impurity_vals:
              best_feature = index
              best_class_impurity_vals, best_split =  copy(temp_class_impurity_vals), copy(temp_split)
              #print(best_class_impurity_vals, best_split)
              best_c_o_d = c_o_d
        else:
          temp_class_impurity_vals, temp_split = self.discrete_split(column, Y, "Gini")
          print(temp_class_impurity_vals)
          if best_class_impurity_vals == None:
              best_feature = index
              best_class_impurity_vals, best_split =  temp_class_impurity_vals, temp_split
              best_c_o_d = c_o_d
              #print(best_split)
          elif best_class_impurity_vals > temp_class_impurity_vals:
              best_feature = index
              best_class_impurity_vals, best_split =  copy(temp_class_impurity_vals), copy(temp_split)
              #print(best_class_impurity_vals, best_split)
              best_c_o_d = c_o_d

    print("BEST VALUES", best_class_impurity_vals, best_split)
    if c_o_d == 'continuous':
      r_subset_Y = Y[X[:,best_feature]>best_split]
      r_subset_X = X[X[:,best_feature]>best_split, :]
      l_subset_Y = Y[X[:,best_feature]<=best_split]
      l_subset_X = X[X[:,best_feature]<=best_split, :]
    else:
      r_subset_Y = Y[X[:,best_feature]!=best_split]
      r_subset_X = X[X[:,best_feature]!=best_split, :]
      l_subset_Y = Y[X[:,best_feature]==best_split]
      l_subset_X = X[X[:,best_feature]==best_split, :]
    if y_c_o_d == 'discrete':
      unique, counts = np.unique(r_subset_Y, return_counts=True)
      max_index = np.argmax(counts)
      most_frequent = unique[max_index]
      r_prediction = most_frequent
      unique, counts = np.unique(l_subset_Y, return_counts=True)
      max_index = np.argmax(counts)
      most_frequent = unique[max_index]
      l_prediction = most_frequent
    else:
      l_prediction = np.average(l_subset_Y)
      r_prediction = np.average(r_subset_Y)

    print(f'Predictions l_pred: {l_prediction} r_pred: {r_prediction}')
    tree.c_o_d = best_c_o_d
    tree.split = best_split
    tree.impurity_val = best_class_impurity_vals
    tree.feature = best_feature
    #print(tree)
    l_node = BTnode(None, None, tree, None, None, None, np.where(X[:,best_feature]<=best_split), pred_val = l_prediction)
    r_node = BTnode(None, None, tree, None, None, None, np.where(X[:,best_feature]>best_split), pred_val = r_prediction)
    r_node = self.find_best_split(r_subset_X, r_subset_Y, depth, r_node, y_c_o_d)
    l_node = self.find_best_split(l_subset_X, l_subset_Y, depth, l_node, y_c_o_d)
    tree.l_child = l_node
    tree.r_child = r_node
    return tree

  def pred_val_from_tree(self, x, curr_node):
    if curr_node.l_child == None and curr_node.r_child == None:
      return curr_node.pred_val
    if curr_node.c_o_d == 'continuous':
      if x[curr_node.feature] > curr_node.split:
        pred_val = self.pred_val_from_tree(x, curr_node.r_child)
      elif x[curr_node.feature] <= curr_node.split:
        pred_val = self.pred_val_from_tree(x, curr_node.l_child)
    if curr_node.c_o_d == 'discrete':
      if x[curr_node.feature] != curr_node.split:
        pred_val = self.pred_val_from_tree(x, curr_node.r_child)
      elif x[curr_node.feature] == curr_node.split:
        pred_val = self.pred_val_from_tree(x, curr_node.l_child)
    return pred_val


  def predict(self, X):
    pred_vals = []
    for row in X:
      pred_vals.append(self.pred_val_from_tree(row, self.trained_dt))
    return pred_vals

