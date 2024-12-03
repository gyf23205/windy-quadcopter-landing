import numpy as np
import matplotlib.pyplot as plt

class WindGen(object):
    def __init__(self, param, dt):
        self.dt = dt
        self.mean = np.array([param[0, 0], param[1, 0]])
        self.std = np.array([param[0, 1], param[1, 1]])
        self.auto = np.array([param[0, 2], param[1, 2]])
        self.now = np.random.normal(self.mean, self.std)
        print(self.now)
        self.alpha = np.exp(-dt / self.auto)

    def generate(self):
        next = self.alpha * self.now + (1 - self.alpha) * self.mean + np.sqrt(1 - self.alpha**2) * np.random.normal(0, self.std)
        self.now = next
        return next
    
if __name__=='__main__':
    [10, 2, 0.5], [1, 0.3, 0.5]
    gen = WindGen(np.array([[10, 2, 0.5], [1, 0.3, 0.5]]), 0.1)
    hist = []
    for i in range(100):
        hist.append(gen.generate())
    plt.plot(hist)
    plt.show()
