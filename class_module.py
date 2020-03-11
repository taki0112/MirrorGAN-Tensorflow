from ops import *
from tensorflow.keras import Sequential


##################################################################################
# Generator
##################################################################################
class CnnEncoder(tf.keras.Model):
    def __init__(self, embed_dim, name='CnnEncoder'):
        super(CnnEncoder, self).__init__(name=name)
        self.embed_dim = embed_dim

        self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        self.inception_v3.trainable = False

        self.inception_v3_mixed7 = tf.keras.Model(inputs=self.inception_v3.input, outputs=self.inception_v3.get_layer('mixed7').output)
        self.inception_v3_mixed7.trainable = False

        self.emb_feature = Conv(channels=self.embed_dim, kernel=1, stride=1, use_bias=False, name='emb_feature_conv') # word_feature
        self.emb_code = FullyConnected(units=self.embed_dim, use_bias=True, name='emb_code_fc') # sent code

    def call(self, x, training=True, mask=None):
        x = ((x + 1) / 2) * 255.0
        x = resize(x, [299, 299])
        x = self.inception_v3_preprocess(x)

        code = self.inception_v3(x)
        feature = self.inception_v3_mixed7(x)

        feature = self.emb_feature(feature)
        code = self.emb_code(code)

        return feature, code

class RnnEncoder(tf.keras.Model):
    def __init__(self, n_words, embed_dim=256, drop_rate=0.5, n_hidden=128, n_layer=1, bidirectional=True, rnn_type='lstm', name='RnnEncoder'):
        super(RnnEncoder, self).__init__(name=name)
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.model = self.architecture()
        self.rnn = VariousRNN(self.n_hidden, self.n_layer, self.drop_rate, self.bidirectional, rnn_type=self.rnn_type, name=self.rnn_type + '_rnn')

    def architecture(self):
        model = []

        model += [EmbedSequence(self.n_words, self.embed_dim, name='embed_layer')] # [bs, seq_len, embed_dim]
        model += [DropOut(self.drop_rate, name='dropout')]

        model = Sequential(model)

        return model


    def call(self, caption, training=True, mask=None):
        # caption = [bs, seq_len]
        x = self.model(caption, training=training)
        word_emb, sent_emb = self.rnn(x, training=training)  # (bs, seq_len, n_hidden * 2) (bs, n_hidden * 2)
        mask = tf.equal(caption, 0)

        # 일단은 mask return 안함 (pytorch)
        # n_hidden * 2 = embed_dim

        return word_emb, sent_emb, mask

class CA_NET(tf.keras.Model):
    def __init__(self, c_dim, name='CA_NET'):
        super(CA_NET, self).__init__(name=name)
        self.c_dim = c_dim # z_dim, condition dimension

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.c_dim * 2, name='mu_fc')]
        model += [Relu()]

        model = Sequential(model)

        return model

    def call(self, sent_emb, training=True, mask=None):
        x = self.model(sent_emb, training=training)

        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]

        c_code = reparametrize(mu, logvar)

        return c_code, mu, logvar

class CaptionCNN(tf.keras.Model):
    def __init__(self, embed_dim, name='CaptionCNN'):
        super(CaptionCNN, self).__init__(name=name)
        self.embed_dim = embed_dim

        self.resnet_152_preprocess = tf.keras.applications.resnet.preprocess_input
        self.resnet_152 = tf.keras.applications.resnet.ResNet152(weights='imagenet', include_top=False, pooling='avg')
        self.resnet_152.trainable = False

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.embed_dim, name='fc')]
        model += [BatchNorm(momentum=0.99, name='batch_norm')]

        model = Sequential(model)

        return model

    def call(self, x, training=True, mask=None):
        x = ((x + 1) / 2) * 255.0
        x = resize(x, [224, 224])
        x = self.resnet_152_preprocess(x)

        feature = self.resnet_152(x)
        feature = self.model(feature, training=training)

        return feature

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, inputs, training=True, mask=None):
    features, hidden = inputs
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CaptionRNN(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=256, n_hidden=512, n_layer=1, name='CaptionRNN'):
        super(CaptionRNN, self).__init__(name=name)
        self.vocab_size = vocab_size # vocab_size = n_words
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.n_layer = n_layer

        self.embedding = EmbedSequence(n_words=self.vocab_size, embed_dim=self.embed_dim, init_range=0.1, name='embed_layer')
        self.rnn = VariousRNN(n_hidden=self.n_hidden, n_layer=self.n_layer, dropout_rate=0.0, bidirectional=False, name='lstm_rnn')
        self.fc = FullyConnected(units=self.vocab_size, name='fc')

    def call(self, inputs, training=True, mask=None):
        feature, caption = inputs
        # caption = caption[:, 1:]
        caption_embedding = self.embedding(caption)
        # feature = tf.tile(tf.expand_dims(feature, axis=1), multiples=[1, caption.shape[1], 1])
        feature = tf.expand_dims(feature, axis=1)
        x = tf.concat([feature, caption_embedding], axis=-1) # [bs, seq_len + 1, embed_dim] [bs, seq_len, vocab_dim]
        x, _ = self.rnn(x, training=training)
        # x = x[:, :-1, :]
        # x = tf.reshape(x, shape=[-1, self.n_hidden])
        x = self.fc(x) # [bs, seq_len + 1, vocab_dim]

        return x

class AttentionNet(tf.keras.layers.Layer):
    def __init__(self, channels, name='AttentionNet'):
        super(AttentionNet, self).__init__(name=name)
        self.channels = channels # idf, x.shape[-1]

        self.word_conv = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='word_conv')
        self.sentence_fc = FullyConnected(units=self.channels, name='sent_fc')
        self.sentence_conv = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='sentence_conv')

    def build(self, input_shape):
        self.bs, self.h, self.w, _ = input_shape[0]
        self.hw = self.h * self.w # length of query
        self.seq_len = input_shape[2][1] # length of source

    def call(self, inputs, training=True):
        x, sentence, context, mask = inputs # context = word_emb
        x = tf.reshape(x, shape=[self.bs, self.hw, -1])

        context = tf.expand_dims(context, axis=1)
        context = self.word_conv(context)
        context = tf.squeeze(context, axis=1)

        attn = tf.matmul(x, context, transpose_b=True) # [bs, hw, seq_len]
        attn = tf.reshape(attn, shape=[self.bs * self.hw, self.seq_len])

        mask = tf.tile(mask, multiples=[self.hw, 1])
        attn = tf.where(tf.equal(mask, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=mask.shape), y=attn)
        attn = tf.nn.softmax(attn)
        attn = tf.reshape(attn, shape=[self.bs, self.hw, self.seq_len])

        weighted_context = tf.matmul(context, attn, transpose_a=True, transpose_b=True)
        weighted_context = tf.reshape(tf.transpose(weighted_context, perm=[0, 2, 1]), shape=[self.bs, self.h, self.w, -1])
        word_attn = tf.reshape(attn, shape=[self.bs, self.h, self.w, -1])

        # Eq(5) in MirrorGAN: global-level attention
        sentence = self.sentence_fc(sentence)
        sentence = tf.reshape(sentence, shape=[self.bs, 1, 1, -1])
        sentence = tf.tile(sentence, multiples=[1, self.h, self.w, 1])

        x = tf.reshape(x, shape=[self.bs, self.h, self.w, -1])

        sent_attn = x * sentence
        sent_attn = self.sentence_conv(sent_attn)
        sent_attn = tf.nn.softmax(sent_attn)

        weighted_sentence = sentence * sent_attn

        return weighted_context, weighted_sentence, word_attn, sent_attn

class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='UpBlock'):
        super(UpBlock, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]

        model = Sequential(model)

        return model

    def call(self, x_init, training=True):
        x = nearest_up_sample(x_init, scale_factor=2)

        x = self.model(x, training=training)

        return x

class Generator_64(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_64'):
        super(Generator_64, self).__init__(name=name)
        self.channels = channels

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.channels * 4 * 4 * 2, use_bias=False, name='code_fc')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]
        model += [tf.keras.layers.Reshape(target_shape=[4, 4, self.channels])]

        for i in range(4):
            model += [UpBlock(self.channels // 2, name='up_block_' + str(i))]
            self.channels = self.channels // 2

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_64_logit')]
        generate_img_block += [Tanh()]

        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block


    def call(self, c_z_code, training=True, mask=None):
        h_code = self.model(c_z_code, training=training)
        x = self.generate_img_block(h_code, training=training)

        return h_code, x


class Generator_128(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_128'):
        super(Generator_128, self).__init__(name=name)
        self.channels = channels # gf_dim

        self.attention_net = AttentionNet(channels=self.channels)
        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels * 2, kernel=1, stride=1, use_bias=False, name='conv')]

        for i in range(2):
            model += [ResBlock(self.channels * 2, name='resblock_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_128_logit')]
        generate_img_block += [Tanh()]

        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block

    def call(self, inputs, training=True):
        h_code, c_code, word_emb, mask = inputs
        c_code, weighted_sentence, _, _ = self.attention_net([h_code, c_code, word_emb, mask])

        h_c_code = tf.concat([h_code, c_code, weighted_sentence], axis=-1)

        h_code = self.model(h_c_code, training=training)
        x = self.generate_img_block(h_code)

        return c_code, h_code, x

class Generator_256(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_256'):
        super(Generator_256, self).__init__(name=name)
        self.channels = channels

        self.attention_net = AttentionNet(self.channels)
        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels * 2, kernel=1, stride=1, use_bias=False, name='conv')]

        for i in range(2):
            model += [ResBlock(self.channels * 2, name='res_block_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]

        model += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_256_logit')]
        model += [Tanh()]

        model = Sequential(model)

        return model

    def call(self, inputs, training=True):
        h_code, c_code, word_emb, mask = inputs
        c_code, weighted_sentence, _, _ = self.attention_net([h_code, c_code, word_emb, mask])

        h_c_code = tf.concat([h_code, c_code, weighted_sentence], axis=-1)

        x = self.model(h_c_code, training=training)

        return x

class Generator(tf.keras.Model):
    def __init__(self, channels, name='Generator'):
        super(Generator, self).__init__(name=name)
        self.channels = channels

        # self.c_dim = c_dim
        # self.ca_net = CA_NET(self.c_dim)

        self.g_64 = Generator_64(self.channels * 16, name='g_64')
        self.g_128 = Generator_128(self.channels, name='g_128')
        self.g_256 = Generator_256(self.channels, name='g_256')

    def call(self, inputs, training=True, mask=None):

        # z_code, sent_emb, word_emb, mask = inputs
        # c_code, mu, logvar = self.ca_net(sent_emb, training=training)

        c_code, z_code, word_emb, mask = inputs
        c_z_code = tf.concat([c_code, z_code], axis=-1)

        h_code1, x_64 = self.g_64(c_z_code, training=training)
        c_code, h_code2, x_128 = self.g_128([h_code1, c_code, word_emb, mask], training=training)
        x_256 = self.g_256([h_code2, c_code, word_emb, mask], training=training)

        x = [x_64, x_128, x_256]

        return x


##################################################################################
# Discriminator
##################################################################################

class DownBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='DownBlock'):
        super(DownBlock, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [Leaky_Relu(alpha=0.2)]

        model = Sequential(model)

        return model

    def call(self, x, training=True):
        x = self.model(x, training=training)

        return x

class Discriminator_64(tf.keras.layers.Layer):
    def __init__(self, channels, name='Discriminator_64'):
        super(Discriminator_64, self).__init__(name=name)
        self.channels = channels # self.df_dim

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='convv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_' + str(i))]
            model += [BatchNorm(name='batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]

        code_block = Sequential(code_block)

        return model, code_block

    def call(self, inputs, training=True):
        x, sent_emb = inputs

        x = self.model(x, training=training)

        # uncondition
        uncond_logit = self.uncond_logit_conv(x)

        # condition
        h_c_code = tf.concat([x, sent_emb], axis=-1)
        h_c_code = self.code_block(h_c_code, training=training)

        cond_logit = self.cond_logit_conv(h_c_code)

        return uncond_logit, cond_logit

class Discriminator_128(tf.keras.layers.Layer):
    def __init__(self, channels, name='Discriminator_128'):
        super(Discriminator_128, self).__init__(name=name)
        self.channels = channels

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()


    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_' + str(i))]
            model += [BatchNorm(name='batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        model += [DownBlock(self.channels * 2, name='down_block')]

        model += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='last_conv')]
        model += [BatchNorm(name='last_batch_norm')]
        model += [Leaky_Relu(alpha=0.2)]

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]

        code_block = Sequential(code_block)

        return model, code_block


    def call(self, inputs, training=True):
        x, sent_emb = inputs

        x = self.model(x, training=training)

        # uncondition
        uncond_logit = self.uncond_logit_conv(x)

        # condition
        h_c_code = tf.concat([x, sent_emb], axis=-1)
        h_c_code = self.code_block(h_c_code, training=training)

        cond_logit = self.cond_logit_conv(h_c_code)

        return uncond_logit, cond_logit

class Discriminator_256(tf.keras.layers.Layer):
    def __init__(self, channels, name='Discriminator_256'):
        super(Discriminator_256, self).__init__(name=name)
        self.channels = channels

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4, use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()


    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_' + str(i))]
            model += [BatchNorm(name='batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        for i in range(2):
            model += [DownBlock(self.channels * 2, name='down_block_' + str(i))]

        for i in range(2):
            model += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='last_conv_' + str(i))]
            model += [BatchNorm(name='last_batch_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]

        code_block = Sequential(code_block)

        return model, code_block


    def call(self, inputs, training=True):
        x, sent_emb = inputs

        x = self.model(x, training=training)

        # uncondition
        uncond_logit = self.uncond_logit_conv(x)

        # condition
        h_c_code = tf.concat([x, sent_emb], axis=-1)
        h_c_code = self.code_block(h_c_code, training=training)

        cond_logit = self.cond_logit_conv(h_c_code)

        return uncond_logit, cond_logit

class Discriminator(tf.keras.Model):
    def __init__(self, channels, embed_dim, name='Discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.channels = channels
        self.embed_dim = embed_dim

        self.d_64 = Discriminator_64(self.channels, name='d_64')
        self.d_128 = Discriminator_128(self.channels, name='d_128')
        self.d_256 = Discriminator_256(self.channels, name='d_256')

    def call(self, inputs, training=True, mask=None):
        x_64, x_128, x_256, sent_emb = inputs
        sent_emb = tf.reshape(sent_emb, shape=[-1, 1, 1, self.embed_dim])
        sent_emb = tf.tile(sent_emb, multiples=[1, 4, 4, 1])

        x_64_uncond_logit, x_64_cond_logit = self.d_64([x_64, sent_emb], training=training)
        x_128_uncond_logit, x_128_cond_logit = self.d_128([x_128, sent_emb], training=training)
        x_256_uncond_logit, x_256_cond_logit = self.d_256([x_256, sent_emb], training=training)

        uncond_logits = [x_64_uncond_logit, x_128_uncond_logit, x_256_uncond_logit]
        cond_logits = [x_64_cond_logit, x_128_cond_logit, x_256_cond_logit]

        return uncond_logits, cond_logits

class Vgg16(tf.keras.Model):
    def __init__(self, name='Vgg16'):
        super(Vgg16, self).__init__(name=name)

        self.vgg_16_preprocess = tf.keras.applications.vgg16.preprocess_input
        self.vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
        self.vgg16.trainable = False

        self.model = tf.keras.Model(inputs=self.vgg16.input, outputs=self.vgg16.get_layer('block4_conv3').output)


    def call(self, x, training=None, mask=None):
        x = ((x + 1) / 2) * 255.0
        x = self.vgg_16_preprocess(x)
        x = self.model(x)

        return x