import autograd.numpy as np

landmarks = np.array([[19.5, 18.5],
                      [18.5, 19.5]])
landmarks1 = np.random.uniform(16.0, 19.0, size=(5, 2))
landmarks2 = np.random.uniform(16.0, 19.0, size=(5, 2))
landmarks = np.concatenate((np.concatenate((landmarks1, landmarks2)), landmarks))

np.save('landmarks.npy', landmarks)
