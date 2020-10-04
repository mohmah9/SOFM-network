import numpy as np
from matplotlib import pyplot as plt

np.random.seed(5)

class SOFM():
    def __init__(self,width=40,height=40,Num_of_colors=1600,lr=0.01,epochs=10000):
        self.data = np.random.randint(0, 255, (3, Num_of_colors))
        # self.network_dimensions = np.array([height, width])
        self.iterations = epochs
        self.initial_learning_rate = lr
        self.features = self.data.shape[0]
        self.count_data = self.data.shape[1]
        self.initial_radius = max(height, width) / 2
        # radius decay parameter
        self.time_constant = self.iterations / np.log(self.initial_radius)
        self.data = self.data / self.data.max()
        ''' 1 initialization '''
        self.net = np.random.random((height, width, self.features))

    # def decrease_radius(self, i):
        # minimum_radius = 1
        # diff = (self.init_radius-minimum_radius)/ self.n_iterations
        # return self.init_radius - (i*diff)

    # def decrease_learning_rate(self, i):
        # minimum_lr = 0.0036
        # diff = (self.init_learning_rate - minimum_lr) / self.n_iterations
        # return self.init_learning_rate - (i * diff)

    def min_finder(self,t, net):
        t = t.reshape(1, 3)
        w = np.sqrt(np.sum((net - t) ** 2, axis=2))
        min_index = np.unravel_index(w.argmin(), w.shape)
        return min_index
    def fit(self):
        for i in range(self.iterations+1):
            if i % 1000 == 0:
                print("epoch", i)
            ''' 2 competition '''
            t = self.data[:, np.random.randint(0, self.count_data)].reshape(np.array([self.features, 1]))
            min_index = self.min_finder(t, self.net)
            # decay the SOM parameters
            r = self.initial_radius * np.exp(-i / self.time_constant)
            lr = self.initial_learning_rate * np.exp(-i / self.iterations)
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = self.net[x, y, :].reshape(self.features, 1)
                    # print(w)
                    distance = np.sum((np.array([x, y]) - min_index) ** 2)
                    distance = np.sqrt(distance)
                    if distance <= r:
                        ''' 3 cooperation '''
                        h = np.exp(-distance / (2 * (r ** 2)))
                        ''' 4 adaptaion '''
                        new_w = w + (lr * h * (t - w))
                        self.net[x, y, :] = new_w.reshape(1, 3)

    def ploting(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Init Data _ 1600 color square')
        ax1.imshow(self.data.reshape(self.net.shape[0], self.net.shape[1], 3))
        ax2.set_title('SOFM result')
        ax2.imshow(self.net)
        plt.show()

my_som=SOFM()
my_som.fit()
my_som.ploting()