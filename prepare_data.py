import pickle
import os
from sklearn.datasets import load_svmlight_file

if not os.path.exists('data'):
    os.makedirs('data')

# 处理 rcv1
X, y = load_svmlight_file('data/rcv1_test.binary.bz2')
with open('data/rcv1.pickle', 'wb') as f:
    pickle.dump((X, y), f, protocol=pickle.HIGHEST_PROTOCOL)

# 处理 epsilon
X, y = load_svmlight_file('data/epsilon_normalized.bz2')
with open('data/epsilon.pickle', 'wb') as f:
    pickle.dump((X, y), f, protocol=pickle.HIGHEST_PROTOCOL)
