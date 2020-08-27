import numpy as np
import keras
import tensorflow as tf

class DataLoader(keras.utils.Sequence):

    def __init__(
        self,
        mode,
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        ):

        """
        Dataloader class
        Parameters:
            * mode: train or test
            * dataset: dataset to load
            * batch_size: how many samples in one batch (default: ``8``)
            * shuffle: shuffle data at every epoch, if True (default: ``True``)
            * num_workers: how many threads to use for data loading in one batch. 0 means that                  the data will be loaded in the main process (default: ``0``)
        """

        self.mode=mode
        self.dataset=dataset
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.indices=[]
        self.on_epoch_end()

    def __getitem__(self,index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        samples = []

        # Test
        if self.mode=='test':
            for i in indices:
                data = self.dataset[i]
                samples.append(data)
            img=np.array([sample for sample in samples])    
            return img, None

        # Train                
        else:
            for i in indices:
                data = self.dataset[i]
                samples.append(data)
            left=np.array([sample[0] for sample in samples])
            right=np.array([sample[1] for sample in samples])
            left_right=tf.concat([left,right],axis=-1)
            return left, left_right
    
    def on_epoch_end(self):
        n=len(self.dataset)
        seq=np.arange(0,n)
        if self.shuffle:
            np.random.shuffle(seq)
        self.indices=seq
    
    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))
