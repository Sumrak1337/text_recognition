import numpy as np


class DataSet:
    """Class for training data and feeding train function."""
    def __init__(self, img, lbl):
        self.images = img
        self.labels = lbl
        self.length = len(img)
        self.index = 0

    def next_batch(self, batch_size):
        """Return the next batch from the data set."""
        start = self.index
        self.index += batch_size

        if self.index > self.length:
            # Shuffle the data
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index = batch_size

        end = self.index
        return self.images[start:end], self.labels[start:end]
