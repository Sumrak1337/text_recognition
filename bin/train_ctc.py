import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_slim as slim
from imgaug import augmenters as iaa
from omegaconf import DictConfig
from tqdm import tqdm

from src.ocr.datahelpers import CHAR_SIZE, char2idx, load_words_data
from src.ocr.helpers import resize
from src.ocr.tfhelpers import create_cell
from src.utils.dataiterator import DataIterator


def train_ctc(cfg: DictConfig):
    sets_root = Path(cfg.csv_path)
    train_images, train_labels = load_words_data(sets_root / "train.csv", is_csv=True)
    dev_images, dev_labels = load_words_data(sets_root / "dev.csv", is_csv=True)
    slider_size = (64, 64)  # Second parameter can be edited
    vocab_size = CHAR_SIZE + 2  # Number of different chars + <PAD> and <EOS>

    os.makedirs(cfg.models_output, exist_ok=True)
    # save_loc += model_name
    data = {
        "train": (
            train_images,
            train_labels,
            np.empty(len(train_labels), dtype=object),
        ),
        "dev": (dev_images, dev_labels, np.empty(len(dev_labels), dtype=object)),
    }

    for d in tqdm(data):
        for i in range(len(data[d][0])):
            data[d][0][i] = resize(data[d][0][i], slider_size[1], True)
            data[d][2][i] = [char2idx(c) for c in data[d][1][i]]

    print("Training images:", len(train_images))
    print("Testing images:", len(dev_images))

    seq = iaa.Sequential(
        [
            iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0.5, 10.0), sigma=5.0)),
            iaa.OneOf(
                [
                    iaa.GaussianBlur((0, 0.5)),
                    iaa.AverageBlur(k=(1, 3)),
                    iaa.MedianBlur(k=(1, 3)),
                ]
            ),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=0.01 * 255)),
        ]
    )
    train_iterator = DataIterator(
        data["train"][0],
        data["train"][2],
        cfg.model_params.num_buckets,
        slider_size,
        augmentation=seq,
        dropout=cfg.model_params.dropout,
        train=True,
    )
    test_iterator = DataIterator(
        data["dev"][0], data["dev"][2], 1, slider_size, train=False
    )

    # Placeholders
    inputs = tf.compat.v1.placeholder(
        shape=(None, slider_size[0], None, 1), dtype=tf.float32, name="inputs"
    )
    inputs_length = tf.compat.v1.placeholder(
        shape=(None,), dtype=tf.int32, name="inputs_length"
    )
    targets = tf.compat.v1.sparse_placeholder(dtype=tf.int32, name="targets")
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")

    # Graph
    # 1. Convulation
    conv1 = slim.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[7, 7],
        strides=(2, 2),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )

    conv12 = slim.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )
    # 2. Max Pool
    pool1 = tf.compat.v1.layers.max_pooling2d(conv12, pool_size=[2, 2], strides=2)
    # 3. Inception
    conv2 = slim.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )

    conv22 = tf.compat.v1.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )
    # 4. Max Pool
    pool2 = tf.compat.v1.layers.max_pooling2d(conv22, pool_size=[2, 1], strides=[2, 1])
    # 5. Inception
    conv3 = tf.compat.v1.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )
    conv32 = tf.compat.v1.layers.conv2d(
        inputs=conv3,
        filters=256,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )
    # 6. Max Pool
    pool3 = tf.compat.v1.layers.max_pooling2d(conv32, pool_size=[2, 2], strides=2)

    # Image patches for RNN
    image_patches1 = tf.compat.v1.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[
            slider_size[0] // cfg.model_params.hn_pool,
            slider_size[1] // cfg.model_params.wn_pool,
        ],
        strides=(1, 1),
        padding="same",
        kernel_initializer=slim.xavier_initializer(),
        kernel_regularizer=slim.l2_regularizer(scale=cfg.model_params.scale),
        activation=tf.nn.relu,
    )

    image_patches = tf.compat.v1.layers.separable_conv2d(
        inputs=image_patches1,
        filters=1280,
        kernel_size=[slider_size[0] // cfg.model_params.hn_pool, 1],
        strides=(1, 1),
        depth_multiplier=5,
        name="image_patches",
    )

    processed_inputs = tf.transpose(
        tf.squeeze(image_patches, [1]), [1, 0, 2], name="squeeze_transpose"
    )
    lengths = inputs_length // cfg.model_params.wn_pool

    # Bi-RNN
    enc_cell_fw = create_cell(
        cfg.model_params.units,
        cfg.model_params.layers,
        cfg.model_params.residual_layers,
        is_dropout=True,
        keep_prob=keep_prob,
    )
    enc_cell_bw = create_cell(
        cfg.model_params.units,
        cfg.model_params.layers,
        cfg.model_params.residual_layers,
        is_dropout=True,
        keep_prob=keep_prob,
    )
    bi_outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
        cell_fw=enc_cell_fw,
        cell_bw=enc_cell_bw,
        inputs=processed_inputs,
        sequence_length=lengths,
        dtype=tf.float32,
        time_major=True,
    )

    con_outputs = tf.concat(bi_outputs, -1)
    logits = tf.compat.v1.layers.dense(inputs=con_outputs, units=vocab_size)

    # Final outputs
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        logits, lengths, merge_repeated=False
    )
    word_prediction = tf.compat.v1.sparse_tensor_to_dense(
        decoded[0], name="word_prediction"
    )

    # Optimizer
    ctc_loss = tf.nn.ctc_loss(
        targets,
        logits,
        lengths,
        time_major=True,
        ctc_merge_repeated=True,
        ignore_longer_outputs_than_inputs=True,
    )
    regularization = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
    )
    loss = tf.identity(tf.reduce_mean(ctc_loss) + sum(regularization), name="loss")

    optimizer = tf.compat.v1.train.AdamOptimizer(cfg.model_params.learning_rate)
    train_step = optimizer.minimize(loss, name="train_step")

    # Label error rate
    label_err_rate = tf.reduce_mean(
        tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
    )

    # Training
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    # TesorBoard stats
    tf.summary.scalar("Loss", loss)
    tf.summary.scalar("Label_err_rate", label_err_rate)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    i_batch = 0
    try:
        for i_batch in tqdm(range(cfg.model_params.train_steps)):
            fd = train_iterator.next_feed(cfg.model_params.batch_size)
            sess.run(train_step, feed_dict=fd)

            if i_batch % cfg.model_params.save_iter == 0:
                saver.save(
                    sess,
                    str(Path(cfg.models_output) / cfg.model_params.model_name),
                    global_step=i_batch,
                )

            if i_batch % cfg.model_params.epoch == 0:
                fd_test = test_iterator.next_feed(cfg.model_params.batch_size)
                print("Batch %r - Loss: %r" % (i_batch, sess.run(loss, fd)))
                print(
                    "    Train Label Err Rate: %r"
                    % sess.run(label_err_rate, feed_dict=fd)
                )
                print(
                    "    Eval Label Err Rate: %r"
                    % sess.run(label_err_rate, feed_dict=fd_test)
                )
                print()

    except KeyboardInterrupt:
        print("Stopped on batch:", i_batch)
        saver.save(sess, str(Path(cfg.models_output) / cfg.model_params.model_name))
        print("Training interrupted, model saved.")
