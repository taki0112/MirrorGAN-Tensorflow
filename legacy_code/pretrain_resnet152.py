import tensorflow as tf

class Resnet_152_feature(tf.keras.Model):
    def __init__(self, class_resnet_152):
        super(Resnet_152_feature, self).__init__(name='Resnet_152_feature')
        self.resnet_152 = class_resnet_152
        self.resnet_152_preprocess = tf.keras.applications.resnet.preprocess_input

    def call(self, x):
        x = ((x + 1) / 2) * 255.0

        x_resnet_152 = self.resnet_152(self.resnet_152_preprocess(x))

        return x_resnet_152

class Resnet_152_class(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Resnet_152_class, self).__init__(name='Resnet_152_class')
        resnet_152_features = tf.keras.applications.resnet.ResNet152(weights='imagenet', include_top=False, pooling='avg')

        if trainable is False:
            resnet_152_features.trainable = False

        self.last_feature = tf.keras.Model(inputs=resnet_152_features.input, outputs=resnet_152_features.output)


    def call(self, x):

        return self.last_feature(x)