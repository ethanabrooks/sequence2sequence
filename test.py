import tensorflow as tf

mem_dim = 4
batch_size = 2
k = tf.ones([batch_size, mem_dim])
# res = k, tf.expand_dims(
#     tf.nn.l2_normalize(k, dim=1), axis=2
#     )
# res = tf.nn.l2_normalize(k, dim=1)
res = tf.split_v(k, [mem_dim / 2, mem_dim / 2], split_dim=1)
print(res)

sess = tf.Session()
print(sess.run(res))
