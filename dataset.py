# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:49
# @Author  : HeJi
# @FileName: dataset.py
# @E-mail: hj@jimhe.cn
import numpy as np
import time
import os
from PIL import Image
import glob
from grammar import select_rect, Grammar
def load_image(img_path, size = (28,28)):
    return Image.open(img_path).resize(size)


def pad_batch(batch_size, X_b, y_b, w_b, ids_b):
    """Pads batch to have size precisely batch_size elements.

    Fills in batch by wrapping around samples till whole batch is filled.
    """
    num_samples = len(X_b)
    if num_samples == batch_size:
        return (X_b, y_b, w_b, ids_b)
    else:
        # By invariant of when this is called, can assume num_samples > 0
        # and num_samples < batch_size
        if len(X_b.shape) > 1:
            feature_shape = X_b.shape[1:]
            X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
        else:
            X_out = np.zeros((batch_size,), dtype=X_b.dtype)

        num_tasks = y_b.shape[1]
        y_out = np.zeros((batch_size, num_tasks), dtype=y_b.dtype)
        w_out = np.zeros((batch_size, num_tasks), dtype=w_b.dtype)
        ids_out = np.zeros((batch_size,), dtype=ids_b.dtype)

        # Fill in batch arrays
        start = 0
        while start < batch_size:
            num_left = batch_size - start
            if num_left < num_samples:
                increment = num_left
            else:
                increment = num_samples
            X_out[start:start + increment] = X_b[:increment]
            y_out[start:start + increment] = y_b[:increment]
            w_out[start:start + increment] = w_b[:increment]
            ids_out[start:start + increment] = ids_b[:increment]
            start += increment
        return (X_out, y_out, w_out, ids_out)


class Dataset(object):
    """Abstract base class for datasets defined by X, y, w elements."""

    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        """
        Get the number of elements in the dataset.
        """
        raise NotImplementedError()

    def get_shape(self):
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        raise NotImplementedError()

    def get_task_names(self):
        """Get the names of the tasks associated with this dataset."""
        raise NotImplementedError()

    @property
    def X(self):
        """Get the X vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    @property
    def y(self):
        """Get the y vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    @property
    def ids(self):
        """Get the ids vector for this dataset as a single numpy array."""

        raise NotImplementedError()

    @property
    def w(self):
        """Get the weight vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def iterbatches(self,
                    batch_size=None,
                    epoch=0,
                    deterministic=False,
                    pad_batches=False):
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
        """
        raise NotImplementedError()

    def itersamples(self):
        """Get an object that iterates over the samples in the dataset.

        Example:

        >>> dataset = NumpyDataset(np.ones((2,2)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x, y, w, id)
        [ 1.  1.] [ 0.] [ 0.] 0
        [ 1.  1.] [ 0.] [ 0.] 1
        """
        raise NotImplementedError()

    def transform(self, fn, **args):
        """Construct a new dataset by applying a transformation to every sample in this dataset.

        The argument is a function that can be called as follows:

        >> newx, newy, neww = fn(x, y, w)

        It might be called only once with the whole dataset, or multiple times with
        different subsets of the data.  Each time it is called, it should transform
        the samples and return the transformed data.

        Parameters
        ----------
        fn: function
          A function to apply to each sample in the dataset

        Returns
        -------
        a newly constructed Dataset object
        """
        raise NotImplementedError()

    def get_statistics(self, X_stats=True, y_stats=True):
        """Compute and return statistics of this dataset."""
        X_means = 0.0
        X_m2 = 0.0
        y_means = 0.0
        y_m2 = 0.0
        n = 0
        for X, y, _, _ in self.itersamples():
            n += 1
            if X_stats:
                dx = X - X_means
                X_means += dx / n
                X_m2 += dx * (X - X_means)
            if y_stats:
                dy = y - y_means
                y_means += dy / n
                y_m2 += dy * (y - y_means)
        if n < 2:
            X_stds = 0.0
            y_stds = 0
        else:
            X_stds = np.sqrt(X_m2 / n)
            y_stds = np.sqrt(y_m2 / n)
        if X_stats and not y_stats:
            return X_means, X_stds
        elif y_stats and not X_stats:
            return y_means, y_stds
        elif X_stats and y_stats:
            return X_means, X_stds, y_means, y_stds
        else:
            return None


class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays."""

    def __init__(self, X = None, y=None, w=None, ids=None):
        if X is not None:
            n_samples = len(X)
        else:
            n_samples = 0
        # The -1 indicates that y will be reshaped to have length -1
        if n_samples > 0:
            if y is not None:
                y = np.reshape(y, (n_samples, -1))
                if w is not None:
                    pass
            else:
                # Set labels to be zero, with zero weights
                y = np.zeros((n_samples, 1))
                # w = np.zeros_like(y)
                w = np.ones_like(y)
        #n_tasks = y.shape[1]
        if ids is None:
            ids = np.arange(n_samples)
        if w is None:
            w = np.ones_like(y)
        self._X = X
        self._y = y
        self._w = w
        self._ids = np.array(ids, dtype=object)
        ids_to_index_map = {}
        init_index = 0
        for the_id in self._ids:
            ids_to_index_map[the_id] = init_index
            init_index += 1
        self.ids_to_index_map = ids_to_index_map
        self.n_samples = n_samples

    def __len__(self):
        """
        Get the number of elements in the dataset.
        """
        return len(self._y)

    def get_shape(self):
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

    def get_task_names(self):
        """Get the names of the tasks associated with this dataset."""
        return np.arange(self._y.shape[1])

    @property
    def X(self):
        """Get the X vector for this dataset as a single numpy array."""
        return self._X

    @property
    def y(self):
        """Get the y vector for this dataset as a single numpy array."""
        return self._y

    @property
    def ids(self):
        """Get the ids vector for this dataset as a single numpy array."""
        return self._ids

    @property
    def w(self):
        """Get the weight vector for this dataset as a single numpy array."""
        return self._w

    def iterbatches(self,
                    batch_size=None,
                    epoch=0,
                    deterministic=False,
                    pad_batches=False):
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
        """

        def iterate(dataset, batch_size, deterministic, pad_batches):
            n_samples = dataset._X.shape[0]
            if not deterministic:
                sample_perm = np.random.permutation(n_samples)
            else:
                sample_perm = np.arange(n_samples)
            if batch_size is None:
                batch_size = n_samples
            interval_points = np.linspace(
                0, n_samples, np.ceil(float(n_samples) / batch_size) + 1, dtype=int)
            for j in range(len(interval_points) - 1):
                indices = range(interval_points[j], interval_points[j + 1])
                perm_indices = sample_perm[indices]
                X_batch = dataset._X[perm_indices]
                y_batch = dataset._y[perm_indices]
                w_batch = dataset._w[perm_indices]
                ids_batch = dataset._ids[perm_indices]
                if pad_batches:
                    (X_batch, y_batch, w_batch, ids_batch) = pad_batch(
                        batch_size, X_batch, y_batch, w_batch, ids_batch)
                yield (X_batch, y_batch, w_batch, ids_batch)

        return iterate(self, batch_size, deterministic, pad_batches)


    def split_train_test(self, frac=0.8):
        critical_point = int(self.X.shape[0] * frac)
        train_indexs = np.random.choice(self.n_samples, (critical_point,), replace=False)
        test_indexs = np.array([i for i in range(self.n_samples) if i not in train_indexs])
        train = NumpyDataset(self._X[train_indexs], self._y[train_indexs],
                             self._w[train_indexs], self._ids[train_indexs])
        test = NumpyDataset(self._X[test_indexs], self._y[test_indexs],
                            self._w[test_indexs], self._ids[test_indexs])

        return train, test

    def sample_by_ids(self, the_ids):
        indexs = np.array([self.ids_to_index_map[i] for i in the_ids])
        sampled_data = NumpyDataset(self._X[indexs], self._y[indexs],
                                    self._w[indexs], self._ids[indexs])
        return sampled_data

    def sample_by_index(self, the_index):
        sampled_data = NumpyDataset(self._X[the_index], self._y[the_index],
                                    self._w[the_index], self._ids[the_index])
        return sampled_data

    def flow_from_directory(self, path, target_size = (64,64)):
        """
        specific for image data
        :param path: the path to image
        :return:
        """
        all_datas = []
        all_labels = np.array([])

        class_count = 0
        for root, _, filenames in os.walk(path):
            if filenames:
                class_name = class_count
                class_data = []
                length = len(filenames)
                label = np.tile([class_name], length)
                for j, im_file in enumerate(filenames):
                    data = np.array(load_image(os.path.join(root, im_file), size=target_size), dtype=np.float32)
                    #print("data", data.shape)
                    data = np.expand_dims(data, axis=0)
                    class_data.append(data)
                class_data = np.concatenate(class_data, axis=0)
                #print(class_data.shape)
                all_datas.append(class_data)
                all_labels = np.concatenate([all_labels, label])
                class_count += 1
        print(all_labels.shape)
        all_datas = np.concatenate(all_datas, axis=0)

        print(all_datas.shape)
        self.__init__(all_datas, all_labels)






def get_task_supports_and_queries(dataset, n_episodes, n_way, n_shot,
                                  n_query, task, log_every_n=50):
    y_task = dataset.y[:, task]
    w_task = dataset.w[:, task]
    print(task)
    # print(y_task,w_task)

    # Split data into pos and neg lists.

    # pos_mols = np.where(np.logical_and(y_task == 1, w_task != 0))[0]
    # neg_mols = np.where(np.logical_and(y_task == 0, w_task != 0))[0]

    all_labels = np.array(list(set(dataset.y.flatten())))
    n_classes = len(all_labels)

    label_indexs = {i: np.where(np.logical_and(y_task == i, w_task != 0))[0]
                    for i in all_labels}

    # print(pos_mols,neg_mols)
    supports = []
    queries = []
    for episode in range(n_episodes):
        if episode % log_every_n == 0:
            print("Sampling support %d" % episode)

        sampled_ids_support = np.array([])
        sampled_ids_query = np.array([])
        # No replacement allowed for supports
        use_labels = all_labels[np.random.permutation(n_classes)[:n_way]]
        for label in use_labels:
            label_index = label_indexs[label]
            selected = np.random.permutation(len(label_index))[:n_shot + n_query]
            sampled_label_index_support = label_index[selected[:n_shot]]
            sampled_label_index_query = label_index[selected[n_shot:]]
            sampled_label_id_support = dataset.ids[sampled_label_index_support]
            sampled_label_id_query = dataset.ids[sampled_label_index_query]
            # pos_inds, neg_inds = pos_mols[pos_ids], neg_mols[neg_ids]
            # Handle one-d vs. non one-d feature matrices
            sampled_ids_support = np.concatenate([sampled_ids_support, sampled_label_id_support])
            sampled_ids_query = np.concatenate(([sampled_ids_query, sampled_label_id_query]))
        yield (dataset.sample_by_ids(sampled_ids_support), dataset.sample_by_ids(sampled_ids_query))
        #np.random.shuffle(sampled_ids_query)
        #supports.append(sampled_ids_support)
        #queries.append(sampled_ids_query)
    #return supports, queries


def simple_data_generator(X, y, batch_size = 24, shuffle = True, till_end = False):
    data_length = len(y)
    indexes = np.array(list(range(data_length)))
    if shuffle:
        np.random.shuffle(indexes)
    num_batches = data_length // batch_size
    for epi in range(num_batches):
        selected = indexes[epi*batch_size:(epi+1)*batch_size]
        X_batch = X[selected]
        y_batch = y[selected]
        yield X_batch, y_batch
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size * num_batches:]
            X_batch = X[selected]
            y_batch = y[selected]
            yield X_batch, y_batch

def zeropad_to_max_len(data, max_len = 121):
    return np.pad(data, [(0, 0), (0, max_len - data.shape[1]), (0,0)], mode="constant")


class Data_Generator(object):

    def __init__(self, hsi, y, use_coords, batch_size = 24,
                 selection_rules = None, shuffle = True,
                 till_end = False, max_len = 121):
        self.hsi = hsi
        self.y = y
        self.use_coords = use_coords
        self.batch_size = batch_size
        self.selection_rules = selection_rules
        self.shuffle = shuffle
        self.till_end = till_end
        self.max_len = max_len
        self.num_batches = len(self.use_coords) // self.batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.use_coords))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_data = len(self.use_coords)
        if num_data % self.batch_size == 0:
            return self.num_batches
        elif self.till_end:
            return self.num_batches+1
        else:
            return self.num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        try:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        except:
            indexes = self.indexes[index*self.batch_size:]

        X_batch, y_batch = self.__data_generation(indexes)

        return X_batch, y_batch


    def __data_generation(self, indexes):
        region = "rect 11"
        coords = self.use_coords[indexes]
        if self.selection_rules is not None:
            region = np.random.choice(self.selection_rules)
        X_batch = Grammar(self.hsi, coords, method=region)
        X_batch_shape = X_batch.shape
        try:
            y_batch = self.y[coords[:, 0], coords[:, 1]]
        except:
            y_batch = self.y[indexes]

        if len(X_batch_shape) == 4:
            X_batch = np.reshape(X_batch, [X_batch_shape[0], X_batch_shape[1] * X_batch_shape[2], X_batch_shape[3]])
            #X_test = np.reshape(X_t

        X_batch = zeropad_to_max_len(X_batch, max_len=self.max_len)
        return X_batch, y_batch


def data_generator_v2(hsi, y, use_coords, batch_size = 24,
                      selection_rules = None, pitch_size = 11,
                      r = 12, length = 10, shuffle = True,
                      till_end = False, max_len = 121):
    data_length = len(use_coords)
    indexes = np.array(list(range(data_length)))
    region_type = "rect"
    if selection_rules is not None:
        region_type = np.random.choice(selection_rules)
    if shuffle:
        np.random.shuffle(indexes)
    num_batches = data_length // batch_size
    for epi in range(num_batches):
        selected = indexes[epi * batch_size:(epi + 1) * batch_size]
        selected_coords = use_coords[selected]
        X_batch = Grammar(hsi, selected_coords, method=region_type,
                          pitch_size=pitch_size, r=r, length=length)
        y_batch = y[selected_coords[:,0], selected_coords[:,1]]
        X_batch = zeropad_to_max_len(X_batch, max_len=max_len)
        yield X_batch, y_batch
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size*num_batches:]
            selected_coords = use_coords[selected]
            X_batch = Grammar(hsi, selected_coords, method=region_type,
                          pitch_size=pitch_size, r=r, length=length)
            y_batch = y[selected_coords[:,0], selected_coords[:,1]]
            X_batch = zeropad_to_max_len(X_batch, max_len=max_len)
            yield X_batch, y_batch

class EpisodeGenerator(object):
  #"TODO (He Ji) we need exclude the test examples from support"
  """Generates (support, test) pairs for episodic training.

  Precomputes all (support, test) pairs at construction. Allows to reduce
  overhead from computation.
  """
  def __init__(self, dataset, n_way, n_shot, n_query, n_episodes_per_task):
    """
    Parameters
    ----------
    dataset: dc.data.Dataset
      Holds dataset from which support sets will be sampled.
    n_pos: int
      Number of positive samples
    n_neg: int
      Number of negative samples.
    n_test: int
      Number of samples in test set.
    n_episodes_per_task: int
      Number of (support, task) pairs to sample per task.
    replace: bool
      Whether to use sampling with or without replacement.
    """
    time_start = time.time()
    self.tasks = range(len(dataset.get_task_names()))
    self.n_tasks = len(self.tasks)
    self.n_episodes_per_task = n_episodes_per_task
    self.dataset = dataset
    self.n = n_shot
    self.n_way = n_way
    self.n_query = n_query
    self.task_episodes = {}

    for task in range(self.n_tasks):
      task_supports, task_tests = get_task_supports_and_queries(dataset,
                                                                self.n_episodes_per_task,
                                                                n_way, n_shot, n_query, task, log_every_n=50)
      self.task_episodes[task] = (task_supports, task_tests)

    # Init the iterator
    self.perm_tasks = np.random.permutation(self.tasks)
    # Set initial iterator state
    self.task_num = 0
    self.trial_num = 0
    time_end = time.time()
    print("Constructing EpisodeGenerator took %s seconds"
          % str(time_end-time_start))

  def __iter__(self):
    return self

  def next(self):
    """Sample next (support, test) pair.

    Return from internal storage.
    """
    if self.trial_num == self.n_episodes_per_task:
      raise StopIteration
    else:
      task = self.perm_tasks[self.task_num]  # Get id from permutation
      #support = self.supports[task][self.trial_num]
      task_supports, task_tests = self.task_episodes[task]
      support, test = (task_supports[self.trial_num],
                       task_tests[self.trial_num])
      # Increment and update logic
      self.task_num += 1
      if self.task_num == self.n_tasks:
        self.task_num = 0  # Reset
        self.perm_tasks = np.random.permutation(self.tasks)  # Permute again
        self.trial_num += 1  # Upgrade trial index

      return (task, support, test)

  __next__ = next # Python 3.X compatibility

