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


landmarks = np.array([[ 5.5, 5.5],
                      [14.5,14.5]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual.npy', landmarks)



landmarks = np.array([[14.5, 5.5],
                      [ 5.5,14.5]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_2.npy', landmarks)


landmarks = np.array([[14.5, 5.5]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_single.npy', landmarks)


landmarks = np.array([[ 2.5, 2.5],
                      [17.5,17.5]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_3.npy', landmarks)

landmarks = np.array([[ 1.5, 1.5],
                      [13.5,13.5]])
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
# landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_4.npy', landmarks)

landmarks = np.array([[ 1.5, 1.5],
                      [13.5,13.5]])
landmarks1 = np.random.uniform(1., 2., size=(4, 2))
landmarks2 = np.random.uniform(13.0, 14.0, size=(4, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_5.npy', landmarks)


landmarks = np.array([[ 2., 2.],
                      [13.,13.]])
landmarks1 = np.random.uniform(2., 4., size=(10, 2))
landmarks2 = np.random.uniform(11.0, 13.0, size=(10, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_6.npy', landmarks)

landmarks = np.array([[  2.0,  2.0],
                      [  2.2,  1.8],
                      [  1.8,  2.2],
                      [ 12.8, 13.2],
                      [ 13.2, 12.8],
                      [ 13.0, 13.0]])
np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_7.npy', landmarks)


landmarks = np.array([[7., 8.],
                      [8., 7.]])
landmarks1 = np.random.uniform(0.5, 14.5, size=(10, 2))
landmarks2 = np.random.uniform(0.5, 14.5, size=(10, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/uniform_2.npy', landmarks)

