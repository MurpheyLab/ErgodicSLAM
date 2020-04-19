import autograd.numpy as np

landmarks = np.array([[9.5, 10.5],
                      [10.5, 9.5]])
landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
landmarks2 = np.random.uniform(2.0, 8.0, size=(10, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/landmarks.npy', landmarks)
