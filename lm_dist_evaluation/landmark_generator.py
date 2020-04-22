import numpy as np

'''
landmarks = np.array([[9.5, 10.5],
                      [10.5, 9.5]])
landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))
'''

'''
# clustered 1
landmarks = np.array([[16.0, 4.0],
                      [12.0, 8.0],
                      [10.0, 10.0]])
landmarks1 = np.c_[np.random.uniform(2, 4, 15), np.random.uniform(16., 18, 15)]
landmarks = np.concatenate((landmarks, landmarks1))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/clustered_1.npy', landmarks)
'''

'''
# uniform 1
landmarks = np.c_[np.random.uniform(1., 19., 15), np.random.uniform(1., 19., 15)]
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/uniform_1.npy', landmarks)
'''

'''
# clustered 2
landmarks = np.array([[16., 4.]])
landmarks1 = np.c_[np.random.uniform(9, 12, 5), np.random.uniform(8, 11, 5)]
landmarks2 = np.c_[np.random.uniform(2, 3, 15), np.random.uniform(17, 18, 15)]
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/clustered_2.npy', landmarks)
'''

'''
# clustered 3
landmarks = np.array([[16., 4.]])
landmarks1 = np.c_[np.random.uniform(11, 16, 5), np.random.uniform(4, 9, 5)]
landmarks2 = np.c_[np.random.uniform(3, 4, 20), np.random.uniform(16, 17, 20)]
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/clustered_3.npy', landmarks)
'''

'''
# clustered 4
# landmarks1 = np.c_[np.random.uniform(14, 18, 5), np.random.uniform(2, 6, 5)]
# landmarks2 = np.c_[np.random.uniform(2, 6, 20), np.random.uniform(14, 18, 20)]
landmarks1 = np.c_[np.linspace(14, 18, 5), np.linspace(2, 6, 5)]
landmarks2 = np.c_[np.linspace(3, 8, 15), np.linspace(12, 17, 15)]
landmarks = np.concatenate((landmarks1, landmarks2))
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/clustered_4.npy', landmarks)
'''

'''
# clustered 5
landmarks1 = np.c_[np.random.uniform(15, 18, 5), np.random.uniform(2, 5, 5)]
landmarks2 = np.c_[np.random.uniform(6, 7, 15), np.random.uniform(13, 14, 15)]
landmarks = np.concatenate((landmarks1, landmarks2))
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/clustered_5.npy', landmarks)
'''


# two landmarks
landmarks = np.array([[3, 3],
                      [7, 7]])
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/two_diag.npy', landmarks)


