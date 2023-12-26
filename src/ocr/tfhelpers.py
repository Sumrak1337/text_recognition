import tensorflow as tf


class Model:
    """Loading and running isolated tf graph."""

    def __init__(self, loc, operation='activation', input_name='x'):
        """
        loc: location of file containing saved model
        operation: name of operation for running the model
        input_name: name of input placeholder
        """
        self.input = input_name + ":0"
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(str(loc) + '.meta', clear_devices=True)
            saver.restore(self.sess, str(loc))
            self.op = self.graph.get_operation_by_name(operation).outputs[0]

    def run(self, data):
        """Run the specified operation on given data."""
        return self.sess.run(self.op, feed_dict={self.input: data})

    def eval_feed(self, feed):
        """Run the specified operation with given feed."""
        return self.sess.run(self.op, feed_dict=feed)

    def run_op(self, op, feed, output=True):
        """Run given operation with the feed."""
        if output:
            return self.sess.run(
                self.graph.get_operation_by_name(op).outputs[0],
                feed_dict=feed)
        else:
            self.sess.run(
                self.graph.get_operation_by_name(op),
                feed_dict=feed)
