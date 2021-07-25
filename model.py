# -*- coding: utf-8 -*-
"""
DeepACE
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import tensorflow as tf

class  _ChannelNorm(tf.keras.layers.Layer):
    def __init__(self, encoded_len, channel_size, **kwargs):
        super(_ChannelNorm, self).__init__(**kwargs)
        self.encoded_len = encoded_len
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape = (1, 1,
                                                               self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "CN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape = (1, 1, 
                                                              self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "CN_beta") 
    def get_config(self):
        config = super(_ChannelNorm, self).get_config().copy()
        config.update({
            'encoded_len': self.encoded_len,
            'channel_size': self.channel_size,
            'dt': self.dt})
        return config 

    def call(self, inputs): 
        E = tf.reshape(
                tf.reduce_mean(inputs, axis=[2], name="E_CN_rm"), [-1, self.encoded_len, 1], name="E_CN_reshape") 
        Var = tf.reshape(tf.reduce_mean((inputs - E)**2,
                                        axis=[2], name="Var_CN_rm"),[-1, self.encoded_len, 1], name="Var_CN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta
    
class  _GlobalNorm(tf.keras.layers.Layer):
    def __init__(self, channel_size, **kwargs):
        super(_GlobalNorm, self).__init__(**kwargs)
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape = (1, 1,
                                                              self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "GN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape = (1, 1, 
                                                              self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "GN_beta")
    
    def get_config(self):
        config = super(_GlobalNorm, self).get_config().copy()
        config.update({
            'channel_size': self.channel_size,
            'dt': self.dt})
        return config 
    
    def call(self, inputs):
        E = tf.reshape(tf.reduce_mean(inputs, axis=[1, 2], name="E_GN_rm"), [-1, 1, 1], name="E_GN_reshape")
        Var = tf.reshape(
            tf.reduce_mean((inputs - E)**2, axis=[1, 2], name="Var_GN_rm"), [-1, 1, 1], name="Var_GN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta

class Rectifier(tf.keras.layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Rectifier, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer=self.initializer,
            name="kernel",
            trainable=True)
        
    def get_config(self):
        base_config = super(Rectifier, self).get_config()
        config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        pos = tf.keras.activations.relu(inputs)
        neg = tf.keras.activations.relu(-inputs)
        concatenated = tf.concat([pos, neg], axis=-1)
        mixed = tf.matmul(concatenated, self.kernel)
        return mixed

class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, L, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.N = N
        self.L = L
            
    def build(self, inputs):
        self.encoder = tf.keras.layers.Conv1D(
                filters = self.N,
                kernel_size = self.L,
                strides = self.L//2,
                activation = Rectifier(),
                padding = "same",
                name = "encode_conv1d")
        
    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({
            'layers': self.encoder,
            'n': self.N,
            'l': self.L})
        return config 
    
    def call(self, inputs, **kwargs):
        encoded_input = self.encoder(inputs=tf.expand_dims(inputs, -1))
        return encoded_input

class TCN(tf.keras.layers.Layer):
    def __init__(self, N, L, B, H, S, P, X, R, causal, skip, duration, sample_rate, GPU, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.S = S
        self.P = P
        self.X = X
        self.R = R
        self.GPU = GPU
        self.skip = skip
        self.causal = causal
        self.duration = duration
        self.sample_rate = sample_rate
                 
        if self.causal:
            self.padding = "causal"
        else:
            self.padding = "same"
        
    def build(self, inputs):
        
        if self.causal:
            self.encoded_len = 4000
            inp_norm  = _ChannelNorm(self.encoded_len, self.N, name = "Input_Channel_Norm")
            frst_norm = _ChannelNorm(self.encoded_len, self.H, name = "First_Channel_Norm")
            scnd_norm = _ChannelNorm(self.encoded_len, self.H, name = "Second_Channel_Norm")
        else:
            inp_norm  = _GlobalNorm(self.N, name = "Input_Global_Norm")
            frst_norm = _GlobalNorm(self.H, name = "First_Global_Norm")
            scnd_norm = _GlobalNorm(self.H, name = "Second_Global_Norm")
        
        layers = {"input_norm": inp_norm,
            "bottleneck": tf.keras.layers.Conv1D(self.B, 1, 1, name = "bottleneck")}
        
        for r in range(self.R):
            for x in range(self.X):
                this_block = "block_{}_{}_".format(r, x)
                
                layers[this_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.H, kernel_size=1, name = "first_1x1_conv")
                
                layers[this_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name = "first_PReLU")
                
                layers[this_block + "first_norm"] = frst_norm
                
                if self.GPU:
                    layers[this_block + "dw_conv"] = tf.keras.layers.Conv1D(self.H, self.P, 
                                                    padding = self.padding, 
                                                    dilation_rate = 2**x, groups=self.H)
                
                layers[this_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name = "second_PReLU")
                
                layers[this_block + "second_norm"] = scnd_norm
                
                layers[this_block + "res_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1, name = "res_1x1_conv") 
                
                layers[this_block + "skip_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.S, kernel_size=1, name = "skip_1x1_conv")
       
        self.lrs = layers
        
    def get_config(self):
        config = super(TCN, self).get_config().copy()
        config.update({
            'layers': self.lrs,
            'n': self.N,
            'l': self.L,
            'b': self.B,
            'h': self.H,
            's': self.S,
            'p': self.P,
            'x': self.X,
            'r': self.R,
            'gpu': self.GPU,
            'c': self.causal,
            'skip': self.skip,
            'sr': self.sample_rate,
            'duration': self.duration,
            'enc_len': self.encoded_len})
        return config 

    def call(self, encoded_input, **kwargs):
        
        self.encoded_len = encoded_input.shape[1]
        
        norm_input = self.lrs["input_norm"](encoded_input)

        block_input = self.lrs["bottleneck"](norm_input)
        
        skip_connections = 0
        
        for r in range(self.R):
            
            for x in range(self.X):
                
                now_block = "block_{}_{}_".format(r, x)

                block_output = self.lrs[now_block + "first_1x1_conv"](block_input)
                
                block_output = self.lrs[now_block + "first_PReLU"](block_output)
                
                block_output = self.lrs[now_block + "first_norm"](block_output)
                
                if self.GPU:
                    
                    block_output = self.lrs[now_block + "dw_conv"](block_output)
                
                block_output = self.lrs[now_block + "second_PReLU"](block_output)

                block_output = self.lrs[now_block + "second_norm"](block_output)
                
                block_output = self.lrs[now_block + "res_1x1_conv"](block_output)
                                
                skip = self.lrs[now_block + "skip_1x1_conv"](block_output)
                
                block_input = block_output + block_input
                
                skip_connections = skip_connections + skip
        
        if self.skip:
            return skip_connections
        else:
            return block_input

class Masker(tf.keras.layers.Layer):
    def __init__(self, N, **kwargs):
        super(Masker, self).__init__(**kwargs)
        self.N = N
        
    def build(self, inputs):   
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1], name = "decode_PReLU")
        self.decode = tf.keras.layers.Conv1D(self.N, 1, 1, name = "1x1_conv_decoder")
        self.activation = tf.keras.activations.softmax

    def get_config(self):
        config = super(Masker, self).get_config().copy()
        config.update({
            'n': self.N,
            'prelu': self.prelu,
            'decode': self.decode,
            'activation': self.activation})
        return config 
    
    def call(self, inputs, encoded_input, **kwargs):       
        inputs = self.prelu(inputs)
        sep_output = self.decode(inputs) 
        mask = self.activation(sep_output)
        return mask * encoded_input

class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def build(self, inputs):  
        self.deconv = tf.keras.layers.Conv1DTranspose(1, 1, activation = "relu",
                                                      name = "1d_deconv")
        
    def get_config(self):
        config = super(Decoder, self).get_config().copy()
        config.update({
            'deconv': self.deconv})
        return config 
   
    def call(self, inputs,  **kwargs):  
        output = self.deconv(inputs)     
        return output
         
class DeepACE():
    def __init__(self, args):   
        self.N = args.N
        self.L = args.L
        self.B = args.B
        self.H = args.H
        self.S = args.S
        self.P = args.P
        self.X = args.X
        self.R = args.R
        self.G = args.G
        self.GPU = args.GPU
        self.skip = args.skip
        self.top = args.topology
        self.causal = args.causal
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        self.model_name = self.top + "_auto_encoder"

        layers = {"encoder": Encoder(self.N, self.L, name = 'Encoder')}
           
        for i in range(22):
            
            this_block = "channel_{}_".format(i)
            
            layers[this_block + "tcn"] = TCN(self.N, self.L, self.B, self.H, self.S, self.P, 
                                                               self.X, self.R,  
                                                               self.causal, self.skip, self.duration, 
                                                               self.sample_rate, self.GPU, name = 'TCN_{}_'.format(i))
    
            layers[this_block + "masker"] = Masker(self.N, name = 'Masker_{}_'.format(i))
    
            layers[this_block + "decoder"] = Decoder(name = 'Decoder_{}_'.format(i))
            
        self.layers = layers

    def call(self):
        
            inp  = tf.keras.Input(shape = (None,), name = "Input")
            
            enc_inp = self.layers["encoder"](inp)
            
            el = {}

            for i in range(22):
                b_in = enc_inp
                b = self.layers['channel_{}_tcn'.format(i)](b_in)
                b = self.layers["channel_{}_masker".format(i)](b, b_in)
                e = self.layers['channel_{}_decoder'.format(i)](b)
                el['channel_{}_output'.format(i)] = e
                                   
            out = tf.clip_by_value(tf.concat([el['channel_0_output'], 
                              el['channel_1_output'], 
                              el['channel_2_output'],
                              el['channel_3_output'],
                              el['channel_4_output'],
                              el['channel_5_output'],
                              el['channel_6_output'],
                              el['channel_7_output'],
                              el['channel_8_output'],
                              el['channel_9_output'],
                              el['channel_10_output'],
                              el['channel_11_output'],
                              el['channel_12_output'],
                              el['channel_13_output'],
                              el['channel_14_output'],
                              el['channel_15_output'],
                              el['channel_16_output'],
                              el['channel_17_output'],
                              el['channel_18_output'],
                              el['channel_19_output'],
                              el['channel_20_output'],
                              el['channel_21_output']], axis = -1), 0, 1)

            model = tf.keras.Model(inputs = inp, 
                                   outputs = out, name = self.model_name)
            
            for i, w in enumerate(model.weights):
                split_name = w.name.split('/')
                if len(split_name) > 1:
                    new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i) + '/' + split_name[-1] 
                    model.weights[i]._handle_name = new_name
                else:
                    new_name = "new" + '_' + str(i) + '/' + "layer_name" + '_' + str(i) + '/' + split_name[-1] 
            
            return model