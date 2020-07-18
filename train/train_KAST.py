import tensorflow as tf
import argparse
import os
from KAST import KAST
import time
import pdb

import numpy as np
import matplotlib.pyplot as plt
import logger
import OxuvaLoader



parser = argparse.ArgumentParser(description='MAST')
# Data options
parser.add_argument('--datapath', default='gs://datasets_ytvos_davis/YT_VOS/',
                    help='Data path for Youtube-VOS')
parser.add_argument('--validpath', default='gs://datasets_ytvos_davis/DAVIS/',
                    help='Data path for Davis')
parser.add_argument('--savepath', type=str, default='results/test',
                    help='Path for checkpoints and logs')
parser.add_argument('--resume', type=str, default=None,
                    help='Checkpoint file to resume')

# Training options
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate')
parser.add_argument('--bsize', type=int, default=12,
                    help='batch size for training (default: 12)')
parser.add_argument('--worker', type=int, default=12,
                    help='number of dataloader threads')
parser.add_argument('--ssize', type=int, default=10,
                    help='number of frame in the sequence')

parser.add_argument('--firstpass', dest='firstpass', action='store_true', help='Use this when this is the first pass')
parser.add_argument('--no-firstpass', dest='firstpass', action='store_false')
parser.set_defaults(tpu=False)

args = parser.parse_args()

def main():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
        model = KAST()
        optimizer = tf.keras.optimizers.Adam(args.lr)
        training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        training_accuracy = tf.keras.metrics.Mean('accuracy', dtype=tf.float32)

    per_replica_batch_size = args.bsize // strategy.num_replicas_in_sync

    def get_dataset():
        pass

    train_dataset = strategy.experimental_distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, is_training=True))


    @tf.function
    def train_step(iterator, model, optimizer):
      """The step function for one training step"""

      def step_fn(inputs):
        """The computation to run on each TPU device."""
        with tf.GradientTape() as tape:
            loss = model.compute_loss(inputs)
            loss = tf.nn.compute_average_loss(loss, global_batch_size=args.bsize)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        training_loss.update_state(loss * strategy.num_replicas_in_sync)
        training_accuracy.update_state(labels, logits)

      strategy.run(step_fn, args=(next(iterator),))


if __name__ == '__main__':
    main()