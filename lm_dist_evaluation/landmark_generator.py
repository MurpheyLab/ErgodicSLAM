import autograd.numpy as np


'''
landmarks = np.array([[9.5, 10.5],
                      [10.5, 9.5]])
landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/landmarks.npy', landmarks)
'''

'''
##########################################

landmarks = np.array([[10., 10.5]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/center_single.npy', landmarks)

##########################################

landmarks = np.array([[9.5, 10.5],
                      [10.5, 9.5]])
landmarks1 = np.random.uniform(2.0, 18.0, size=(10, 2))
landmarks2 = np.random.uniform(2.0, 18.0, size=(10, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/uniform_1.npy', landmarks)

##########################################

# landmarks = np.array([[9.5, 10.5],
#                       [10.5, 9.5]])
landmarks1 = np.random.uniform(8.0, 12.0, size=(5, 2))
landmarks2 = np.random.uniform(8.0, 12.0, size=(5, 2))
landmarks = np.concatenate((landmarks1, landmarks2))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/centered_1.npy', landmarks)

##########################################

landmarks = np.array([[9.5, 10.5],
                      [10.5, 9.5]])
landmarks1 = np.c_[np.random.uniform(12.5, 16.5, 10), np.random.uniform(3.5, 7.5, 10)]
landmarks2 = np.c_[np.random.uniform(4.5, 8.5, 10), np.random.uniform(13.5, 17.5, 10)]
landmarks = np.concatenate((landmarks1, landmarks2))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_1.npy', landmarks)
'''

landmarks = np.array([[14., 6.],
                      [ 6.,14.]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual.npy', landmarks)


