from pathlib import Path
import tensorflow as tf
import numpy as np
import os
import datetime
import time

from src.cwi_cnn import CWI_CNN
from sklearn.metrics import f1_score
from src.configurations import DefaultConfigurations, Language
from src.paths import REPO_DIR
from src.preprocessor import Preprocessor
from src.evaluate import evaluate
from src import helper

# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
# tf.disable_v2_behavior()
# tf.logging.set_verbosity(tf.logging.ERROR)


# tf.disable_v2_behavior()

@helper.print_execution_time
def run_training(out_dir, x_train, y_train, x_valid, y_valid, features_args, lang, configs):

    # print('Loading datasets from ./resources/dumps/...')
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)

    
    helper.write_lines(features_args, Path(out_dir) / 'features.txt')
    print("Writing to {}\n".format(out_dir))

    print("Generating graph and starting training...")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=configs.ALLOW_SOFT_PLACEMENT,
            log_device_placement=configs.LOG_DEVICE_PLACEMENT
        )
        # session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cwi = CWI_CNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embedding_dims=x_train.shape[2],  # FLAGS.MATRIX_DIM,
                filter_sizes=list(map(int, configs.FILTER_SIZES.split(","))),
                num_filters=configs.NUM_FILTERS,
                l2_reg_lambda=configs.L2_REG_LAMBDA,
                lang=lang
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(configs.LEARNING_RATE)
            # optimizer = tf.train.AdadeltaOptimizer(FLAGS.LEARNING_RATE)

            grads_and_vars = optimizer.compute_gradients(cwi.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cwi.loss)
            acc_summary = tf.summary.scalar("accuracy", cwi.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=configs.NUM_CHECKPOINTS)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, epoch):
                """
                A single training step
                """
                feed_dict = {
                    cwi.input_x: x_batch,
                    cwi.input_y: y_batch,
                    cwi.dropout_keep_prob: configs.DROPOUT_KEEP_PROB
                }
                # print("input_x[0]", x_batch[0])
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cwi.loss, cwi.accuracy], feed_dict
                )
                if step % 10 == 0:
                    print("epoch {}, step {}, loss {:g}, accuracy {:g}".format(epoch, step, loss, accuracy))

                # print("predictions: ", predictions)
                if configs.SAVE_SUMMARIES:
                    train_summary_writer.add_summary(summaries, step)

                # step, loss, accuracy = sess.run([global_step, cwi.loss, cwi.accuracy], feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, accuracy {:g}". format(time_str, step, loss, accuracy))

            def val_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                global best_accuracy

                feed_dict = {
                    cwi.input_x: x_batch,
                    cwi.input_y: y_batch,
                    cwi.dropout_keep_prob: 1.0
                }
                # print("input_x_val[0]", x_batch[0])
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cwi.loss, cwi.accuracy], feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

                # Save checkpoint
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # Generate batches
            # batches = helper.batch_iter(list(zip(x_train, y_train)), configs.BATCH_SIZE, configs.NUM_EPOCHS)

            early_stopping_count = 0
            val_score = 0.0
            # Training loop. For each batch...
            best_model = None
            val_score_str = ""

            data = np.array(list(zip(x_train, y_train)), dtype=object)
            data_size = len(data)
            n_batches_per_epoch = int((data_size - 1) / configs.BATCH_SIZE) + 1

            for epoch in range(configs.NUM_EPOCHS):
                for batch_num in range(n_batches_per_epoch):
                    start_index = batch_num * configs.BATCH_SIZE
                    end_index = min((batch_num + 1) * configs.BATCH_SIZE, data_size)
                    batch = data[start_index:end_index]

                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch, epoch)
                    current_step = tf.train.global_step(sess, global_step)


                    if current_step % configs.EVALUATE_EVERY == 0:
                        # print("\nEvaluation:")
                        x_batch_val, y_batch_val = x_valid, y_valid

                        # val_step(x_batch_val, y_batch_val, writer=dev_summary_writer)
                        feed_dict = {
                            cwi.input_x: x_batch_val,
                            cwi.input_y: y_batch_val,
                            cwi.dropout_keep_prob: 1.0
                        }
                        # print("input_x_val[0]", x_batch[0])
                        step, summaries, loss, accuracy, predictions, out = sess.run(
                            [global_step, dev_summary_op, cwi.loss, cwi.accuracy, cwi.predictions, cwi.output], feed_dict
                        )

                        score = f1_score(np.argmax(y_batch_val, axis=1), predictions, average='macro')

                        # print(out)
                        # time_str = datetime.datetime.now().isoformat()
                        print("\nEvaluation: epoch {} step {}, loss {:g}, accuracy {:g}, f1_score {:g}".format(epoch, step, loss, accuracy,
                                                                                                    score))
                        if configs.SAVE_SUMMARIES and dev_summary_writer:
                            dev_summary_writer.add_summary(summaries, step)

                        # save validation scores
                        if val_score < score:
                            val_score_str += str(score) + "\n"

                        if (val_score <= score or val_score == 0):
                            early_stopping_count = 0
                            val_score = score
                            best_model = sess
                            # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            # print("Saved model checkpoint to {}".format(path))

                        else:
                            early_stopping_count += 1

                        # if early_stopping_count >= 1000:
                        #     break  # early stop if not improved after n iterations

                        # else:
                        #     early_stopping_count += 1
                        # if early_stopping_count >= 100:
                        #     break
                    # if current_step % FLAGS.num_checkpoints == 0:
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))

            # save the best model to disk
            # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            path = saver.save(best_model, checkpoint_prefix, global_step=current_step)
            with open(out_dir / "val_scores.txt", "a") as scores_file:
                scores_file.write(val_score_str)
            print("Saved model checkpoint to {}".format(path))
    return out_dir.stem


def train(x_train, y_train, x_valid, y_valid, features_args, lang=Language.FRENCH, configs=DefaultConfigurations()):
    # Define directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = REPO_DIR / f'models/CNN/{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    with helper.log_stdout(out_dir / 'logs.txt'):
        return run_training(out_dir, x_train, y_train, x_valid, y_valid, features_args, lang, configs)

def train_and_evaluate(features, n_seed=1):
    preprocessor = Preprocessor(features)
    x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents = preprocessor.load_preprocessed_data(seed=42+n_seed)
    model_dir = train(x_train, y_train, x_valid, y_valid, features)
    evaluate(x_test, y_test, x_test_sents, model_dir, features)


def train_and_evaluate_n_times(features, n=5):
    while True:
        nb_train = helper.count_training(features, 'CNN')
        if nb_train >= n:
            print(f'You have trained {nb_train} time. If you want to train more, increase the value n in scripts/train_cnn.py  or delete old trained models.')
            break
        
        print(f'Training: {nb_train+1}/{n}')
        train_and_evaluate(features, n-nb_train)



def main(_):
    train(DefaultConfigurations(Language.FRENCH))


if __name__ == '__main__':
    tf.app.run()
