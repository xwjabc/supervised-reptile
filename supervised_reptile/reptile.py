"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import random

import tensorflow as tf

from .variables import (dot_vars, interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)

import numpy as np

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.  FIXME: Understand the transductive setting here.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op  # FIXME: What is the pre_step_op?

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size,
                   gradagree,
                   sum_w_i_g_i):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
          gradagree: enable gradient agreement.
          sum_w_i_g_i: sum w_i*g_i instead of average them.
        """
        old_vars = self._model_state.export_variables()
        new_vars = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                inputs, labels = zip(*batch)  # Interesting use of zip().
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)  # Restore to old_vars (old state).

        if gradagree is False:
            new_vars = average_vars(new_vars)  # Average over all new_vars theta^\tilde_i.
            self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))
            #   old_vars + meta_step_size * (new_vars - old_vars)
            # = (1 - meta_step_size) * old_vars + meta_step_size * new_vars
        else:
            # I. Naive implementation.
            # sum_j_in_T_gi_gj_list = []
            # for i in range(meta_batch_size):
            #     sum_j_in_T_gi_gj = 0
            #     for j in range(meta_batch_size):
            #         sum_j_in_T_gi_gj += dot_vars(subtract_vars(new_vars[i], old_vars),
            #                                      subtract_vars(new_vars[j], old_vars))
            #         # FIXME: First sum then multiply may get better performance.
            #     sum_j_in_T_gi_gj_list.append(sum_j_in_T_gi_gj)
            # denominator = np.sum(np.abs(np.array(sum_j_in_T_gi_gj_list)))
            # w_i_list = []
            # for i in range(meta_batch_size):
            #     w_i_list[i] = sum_j_in_T_gi_gj_list[i] / denominator

            # II. Slight better implementation.
            g_i_list = []
            for i in range(meta_batch_size):
                g_i_list.append(subtract_vars(old_vars, new_vars[i]))  # g_i = theta - theta_i
            g_avg = average_vars(g_i_list)                             # g_avg
            g_i_g_avg_list = []
            for i in range(meta_batch_size):
                g_i_g_avg_list.append(dot_vars(g_i_list[i], g_avg))    # g_i.dot(g_avg)
            denominator = np.sum(np.abs(g_i_g_avg_list))               # denominator = sum_i |g_i.dot(g_avg)|
            w_i_list = []
            w_i_g_i_list = []
            for i in range(meta_batch_size):
                w_i = g_i_g_avg_list[i] / denominator                  # w_i = g_i.dot(g_avg) / denominator
                w_i_list.append(w_i)
                w_i_g_i_list.append(scale_vars(g_i_list[i], w_i))      # w_i * g_i
            w_i_g_i_avg = average_vars(w_i_g_i_list)                   # FIXME: Multiply with meta_step_size or not?
            if sum_w_i_g_i is False:
                self._model_state.import_variables(subtract_vars(old_vars, scale_vars(w_i_g_i_avg,
                                                                                      meta_step_size)))
            else:
                self._model_state.import_variables(subtract_vars(old_vars, scale_vars(w_i_g_i_avg,  # FIXME: Alternative.
                                                                                    meta_step_size * meta_batch_size)))

            print('denominator: {}, w_i_max: {}, w_i_min: {}'.format(denominator, max(w_i_list), min(w_i_list)))

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        train_set, test_set = _split_train_test(
            _sample_mini_dataset(dataset, num_classes, num_shots+1))  # Add one example for test.
        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            # FIXME: I wonder if information leak really happens? If it is, it means the batch normalization
            # FIXME: is not set as evaluation mode during test prediction? Need to take a deeper look at it.
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)  # test mini-batch = train set + one test example (to avoid info. leak).
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res

class FOML(Reptile):
    """
    A basic implementation of "first-order MAML" (FOML).

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """
    def __init__(self, *args, tail_shots=None, **kwargs):
        """
        Create a first-order MAML session.

        Args:
          args: args for Reptile.
          tail_shots: if specified, this is the number of
            examples per class to reserve for the final
            mini-batch. FIXME: Why are there examples for the final mini-batch?
          kwargs: kwargs for Reptile.
        """
        super(FOML, self).__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            mini_batches = self._mini_batches(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)
            for batch in mini_batches:
                inputs, labels = zip(*batch)
                last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch
        yield tail

def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)  # Allow replacement.
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:  # Make the mini-batch have unique samples.
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)  # Get set of labels.
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break  # Get one example, move it from training set to test set, and then break.
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
