import tensorflow as tf

class CWI_CNN(object):

    def __init__(self, sequence_length, num_classes, embedding_dims, filter_sizes, num_filters,
                 l2_reg_lambda=0.0, lang='FR'):
        self.lang = lang
        print("Loading model...")
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_dims], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


        # Embedding layer
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            print("Training with GPU!")
            with tf.device('/device:GPU:0'), tf.variable_scope("text-embedding"):
                self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)
        else:    
            print("Training with CPU!")
            with tf.device('/cpu:0'), tf.variable_scope("text-embedding"):
                self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)
        #
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_dims, 1, num_filters]
                W = tf.Variable(tf.random_normal(filter_shape), name="W")
                b = tf.Variable(tf.random_normal([num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinarity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")

                # h_drop = tf.nn.dropout(pooled, self.dropout_keep_prob)
                # print("pooled: ", pooled.shape)
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, len(filter_sizes))
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])



        # Final scores and predictions
        with tf.name_scope("output"):

            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.dense = tf.contrib.layers.fully_connected(
                inputs=self.h_drop,
                num_outputs=256,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.random_normal_initializer(),
                trainable=False
            )
            self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)
            self.dense = tf.contrib.layers.fully_connected(
                inputs=self.h_drop,
                num_outputs=64,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.random_normal_initializer(),
                trainable=False
            )
            self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)
            self.output = tf.contrib.layers.fully_connected(
                inputs=self.h_drop,
                num_outputs=num_classes,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.random_normal_initializer(),
                trainable=False
            )

            # Model-Predict
            self.predictions = tf.argmax(tf.nn.softmax(self.output), 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y, logits=self.output, pos_weight=1.7)


            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
