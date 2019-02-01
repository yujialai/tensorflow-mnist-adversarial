import tensorflow as tf
tf.reset_default_graph()
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/tmp/outputyujinet.ckpt")
    print("model restored.")
    print("a value: %s" % W_fc2.eval())
