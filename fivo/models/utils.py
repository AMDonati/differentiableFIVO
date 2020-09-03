import tensorflow as tf


class MLP(tf.keras.Model):

    def __init__(self, layer_sizes, activation_fn, name):
        super(MLP, self).__init__()
        #self.num_layers = len(layer_sizes)
        self.layers_ = [tf.keras.layers.Dense(l, activation=activation_fn, name=name+'_{}'.format(i)) for i,l in enumerate(layer_sizes)]
        self.output_size = layer_sizes[-1]

    def call(self, inputs):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        return x

if __name__ == '__main__':
    mlp = MLP([32,64], activation_fn=tf.nn.relu, name="temp_mlp")
    inputs = tf.ones(shape=(128, 128))
    outputs = mlp(inputs)
    print(outputs.shape)

