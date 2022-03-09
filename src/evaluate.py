import numpy as np
import tensorflow.compat.v1 as tf
from src import helper
from pathlib import Path
from src.paths import REPO_DIR
from src.configurations import Language

# tf.disable_v2_behavior()
# tf.logging.set_verbosity(tf.logging.ERROR)


def evaluate_with_a_checkpoint(x_test, y_test, x_test_sents, checkpoint_dir, features, lang):

    print("======================================================================")
    y_test = np.argmax(y_test, axis=1)

    # checkpoint directory from training run
    load_checkpoint_dir = checkpoint_dir / "checkpoints"
    print("Loading graph from {}".format(load_checkpoint_dir))

    batch_size = 128

    # print("load_checkpoint_dir", load_checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(load_checkpoint_dir)
    print("checkpoint file: ", checkpoint_file)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = helper.batch_iter(list(x_test), batch_size, 1)

            # Collect the prediction scores here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                # print(batch_predictions)
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                # print(all_predictions)

    # Print accuracy if y_test is defined
    helper.evaluation_report(all_predictions, y_test, x_test_sents, checkpoint_dir, features)


def evaluate(x_test, y_test, x_test_sents, model_dir=None, features=None, lang=Language.FRENCH):
    
    p = Path(REPO_DIR / f'models/CNN')
    dirs = sorted(p.iterdir(), key=lambda f: f.stat().st_mtime)

    if len(dirs) > 0:
        if model_dir:
            checkpoint_dir = REPO_DIR / f'models/CNN/{model_dir}'
            print(f"Checkpoint dir: {checkpoint_dir}")
            evaluate_with_a_checkpoint(x_test, y_test, x_test_sents, checkpoint_dir, features, lang)
        else:
            checkpoint_dir = str(dirs[-1])
            print(f"Checkpoint dir: {checkpoint_dir}")
            evaluate_with_a_checkpoint(x_test, y_test, x_test_sents, Path(checkpoint_dir), features, lang)
        return 
    else:
        print("You haven't trained a model yet.")
        print("Run training script to train a model, e.g.,: python scripts/train_all.py")

