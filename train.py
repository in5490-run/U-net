import argparse
import glob
import os
import time

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import unet

# Image size, must conform to bit-sizes.
HEIGHT = 304
WIDTH = 304

# Number of segmentation classes
SEGMENTATION_CLASSES = 4


def generator_for_filenames(*filenames):
    """
    Wrapping a list of filenames as a generator function
    """

    def generator():
        for f in zip(*filenames):
            yield f

    return generator


def preprocess(image, segmentation):
    """
    A preprocess function that is run after images are read.
    Includes som data augmentations.
    """

    # Set images size to a constant
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    segmentation = tf.image.resize(segmentation, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Augmentation
    # Flip left-right
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        segmentation = tf.image.flip_left_right(segmentation)

    # Flip up-down
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        segmentation = tf.image.flip_up_down(segmentation)

    # Rotate
    if tf.random.uniform(()) > 0.5:
        rot_deg = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        tf.image.rot90(image, rot_deg)
        tf.image.rot90(segmentation, rot_deg)

    image = tf.cast(image, tf.float32) / 255
    segmentation = tf.cast(segmentation, tf.int64)

    return image, segmentation


def read_image_and_segmentation(img_f, seg_f):
    """
    Read images from file using tensorflow and convert the segmentation to appropriate format.
    :param img_f: filename for image
    :param seg_f: filename for segmentation
    :return: Image and segmentation tensors
    """
    img_reader = tf.io.read_file(img_f)
    seg_reader = tf.io.read_file(seg_f)
    img = tf.image.decode_png(img_reader, channels=3)
    seg = tf.image.decode_png(seg_reader)[:, :, 1:2]

    # Extract segmentation classes
    water = tf.where(seg == 127, tf.ones_like(seg), tf.zeros_like(seg))
    buildings = tf.where(seg == 33, tf.ones_like(seg), tf.zeros_like(seg))
    roads = tf.where(seg == 76, tf.ones_like(seg), tf.zeros_like(seg))
    void = tf.where(seg == 255, tf.ones_like(seg), tf.zeros_like(seg))
    seg = tf.concat([buildings, roads, water, void], 2)

    return img, seg


def dataset_from_filenames(image_names, segmentation_names, preprocess=preprocess, batch_size=8, shuffle=True):
    """
    Convert a list of filenames to tensorflow images.
    :param image_names: image filenames
    :param segmentation_names: segmentation filenames
    :param preprocess: A function that is run after the images are read, the takes image and
    segmentation as input
    :param batch_size: The batch size returned from the function
    :return: Tensors with images and corresponding segmentations
    """
    dataset = tf.data.Dataset.from_generator(
        generator_for_filenames(image_names, segmentation_names),
        output_types=(tf.string, tf.string),
        output_shapes=(None, None)
    )

    if (shuffle):
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(read_image_and_segmentation)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)

    return dataset


def image_filenames(dataset_folder, training=True):
    """
    Gets filenames for images and segmentation images
    :param dataset_folder: Folder where dataset is located
    :param training: True if training, false if testing/validation
    :return: Two lists for image names and segmentation image names
    """
    sub_dataset = 'training' if training else 'testing'
    segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'y', '*y*.png'),
                                   recursive=True)
    image_names = [f.replace('y', 'x') for f in segmentation_names]
    return image_names, segmentation_names


def vis_mask(image, mask, alpha=0.5):
    """Visualize mask on top of image, blend using 'alpha'."""

    # Images are normalized, 1 is max-value
    buildings = mask[:, :, :, 0:1]
    roads = mask[:, :, :, 1:2]
    water = mask[:, :, :, 2:3]

    red = tf.zeros_like(image) + tf.constant([1, 0, 0], dtype=tf.float32)
    green = tf.zeros_like(image) + tf.constant([0, 1, 0], dtype=tf.float32)
    blue = tf.zeros_like(image) + tf.constant([0, 0, 1], dtype=tf.float32)

    vis = tf.where(buildings, alpha * image + (1 - alpha) * red, image)
    vis = tf.where(roads, alpha * image + (1 - alpha) * green, vis)
    vis = tf.where(water, alpha * image + (1 - alpha) * blue, vis)

    return vis


def main(train_dir):
    # Hyper-parameters
    train_epochs = 100
    batch_size = 4
    learning_rate = 1e-4
    beta_1 = 0.9

    # Model folder and names
    model_name = "{}px_{}py_{}e_{}b_{}lr_{}b1".format(HEIGHT, WIDTH, train_epochs, batch_size, learning_rate, beta_1)
    model_file_name = "{}.h5".format(model_name)
    model_dir = os.path.join(train_dir, model_name)

    # Getting filenames from the dataset
    image_names, segmentation_names = image_filenames('data')

    # Divide into train and test set.
    len_data = len(image_names)
    train_start_idx, train_end_idx = (0, len_data // 100 * 80)
    val_start_idx, val_end_idx = (320, len_data - 1)

    preprocess_train = preprocess
    preprocess_val = preprocess

    # Get image tensors from the filenames
    train_set = dataset_from_filenames(
        image_names[train_start_idx:train_end_idx],
        segmentation_names[train_start_idx:train_end_idx],
        preprocess=preprocess_train,
        batch_size=batch_size
    )
    # Get the validation tensors
    val_set = dataset_from_filenames(
        image_names[val_start_idx:val_end_idx],
        segmentation_names[val_start_idx:val_end_idx],
        batch_size=batch_size,
        preprocess=preprocess_val,
        shuffle=False
    )

    model = unet.unet((HEIGHT, WIDTH, 3), SEGMENTATION_CLASSES)

    loss_fn = losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(lr=learning_rate, beta_1=beta_1)

    print("Summaries are written to '%s'." % model_dir)
    writer = tf.summary.create_file_writer(model_dir, flush_millis=3000)
    summary_interval = 10

    train_loss = metrics.Mean()
    train_iou = metrics.MeanIoU(num_classes=SEGMENTATION_CLASSES)
    train_precision = metrics.Precision()
    train_recall = metrics.Recall()
    train_accuracy = metrics.CategoricalAccuracy()

    val_loss = metrics.Mean()
    val_iou = metrics.MeanIoU(num_classes=SEGMENTATION_CLASSES)
    val_precision = metrics.Precision()
    val_recall = metrics.Recall()
    val_accuracy = metrics.CategoricalAccuracy()

    step = 0
    start_training = start = time.time()
    for epoch in range(train_epochs):

        print("Training epoch: %d" % epoch)
        for image, y in train_set:
            with tf.GradientTape() as tape:
                y_pred = model(image)
                loss = loss_fn(y, y_pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update metrics and step
            train_loss.update_state(loss)
            train_iou.update_state(y, y_pred)
            train_precision.update_state(y, y_pred)
            train_recall.update_state(y, y_pred)
            train_accuracy.update_state(y, y_pred)
            step += 1

            activation = 1 / SEGMENTATION_CLASSES
            if step % summary_interval == 0:
                duration = time.time() - start
                print("step %d. sec/batch: %g. Train loss: %g" % (
                    step, duration / summary_interval, train_loss.result().numpy()))
                # Write summaries to TensorBoard
                with writer.as_default():
                    tf.summary.scalar("train_loss", train_loss.result(), step=step)
                    tf.summary.scalar("train_iou", train_iou.result(), step=step)
                    tf.summary.scalar("train_precision", train_precision.result(), step=step)
                    tf.summary.scalar("train_recall", train_recall.result(), step=step)
                    tf.summary.scalar("train_accuracy", train_accuracy.result(), step=step)
                    vis = vis_mask(image, y_pred >= activation)
                    tf.summary.image("train_image", vis, step=step)

                # Reset metrics and time
                train_loss.reset_states()
                train_iou.reset_states()
                train_precision.reset_states()
                train_recall.reset_states()
                train_accuracy.reset_states()
                start = time.time()

        # Do validation after each epoch
        for i, (image, y) in enumerate(val_set):
            y_pred = model(image)
            loss = loss_fn(y, y_pred)
            val_loss.update_state(loss)
            val_iou.update_state(y, y_pred)
            val_precision.update_state(y, y_pred)
            val_recall.update_state(y, y_pred)
            val_accuracy.update_state(y, y_pred)

            with writer.as_default():
                vis = vis_mask(image, y_pred >= activation)
                tf.summary.image("val_image_batch_%d" % i, vis, step=step, max_outputs=batch_size)

        with writer.as_default():
            tf.summary.scalar("val_loss", val_loss.result(), step=step)
            tf.summary.scalar("val_iou", val_iou.result(), step=step)
            tf.summary.scalar("val_precision", val_precision.result(), step=step)
            tf.summary.scalar("val_recall", val_recall.result(), step=step)
            tf.summary.scalar("val_accuracy", val_accuracy.result(), step=step)
        val_loss.reset_states()
        val_iou.reset_states()
        val_precision.reset_states()
        val_recall.reset_states()
        val_accuracy.reset_states()

    print("Finished training %d epochs in %g minutes." % (
        train_epochs, (time.time() - start_training) / 60))
    # save a model which we can later load by tf.keras.models.load_model(model_path)
    model_path = os.path.join(model_dir, model_file_name)
    print("Saving model to '%s'." % model_path)
    model.save(model_path)
    print(model.summary())


def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Train segmention model on dataset.")
    parser.add_argument("train_dir", help="Directory to put logs and saved model.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args.train_dir)
