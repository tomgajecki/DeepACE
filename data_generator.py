# -*- coding: utf-8 -*-
"""
DeepACE
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import os
import sys
import glob
import librosa
from tqdm import tqdm
import scipy.io as sio
import tensorflow as tf

class DataGenerator():
    def __init__(self, mode, args):
        if mode != "train" and mode != "valid":
            raise ValueError("mode: {} while mode should be "
                             "'train'".format(mode))
        
        if not os.path.isdir(args.data_dir):
            raise ValueError("cannot find data_dir: {}".format(args.data_dir))

        self.wav_dir = os.path.join(args.data_dir, mode)
        self.tfr = os.path.join(args.data_dir, mode + '.tfr')
        self.mode = mode
        self.batch_size = args.batch_size
        self.sample_rate = args.sample_rate
        self.duration = args.duration

        if not os.path.isfile(self.tfr):
            self._encode(self.mode)

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
       
    def fetch(self):
        dataset = tf.data.TFRecordDataset(self.tfr).map(self._decode,
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.mode == "train":
            dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)
            train_dataset = dataset.batch(self.batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return train_dataset
        
        if self.mode == "valid":
            valid_dataset = dataset.batch(1, drop_remainder=True)
            valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE) 
            return valid_dataset

    def _encode(self, mode):
        
        if self.mode == "train":
            print("\nSerializing training data...\n")
        
        if self.mode == "valid":
            print("\nSerializing validation data...\n")

        writer = tf.io.TFRecordWriter(self.tfr)
        
        mix_filenames = glob.glob(os.path.join(self.wav_dir, "*_mixed.wav"))
        target_filenames = glob.glob(os.path.join(self.wav_dir, "*_enhanced.mat"))
        sys.stdout.flush()  
        for mix_filename, target_filename in tqdm(zip(mix_filenames, 
                                                      target_filenames), total = len(mix_filenames)):
            mix, _ = librosa.load(mix_filename, self.sample_rate, mono = True)
            clean = sio.loadmat(target_filename)['lgf_speech']
            clean = clean.astype(mix.dtype)

            def write_audio(a, b):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "noisy" : self._float_list_feature(mix[a:b]),
                            "clean" : self._float_list_feature(clean[:][0:4000].flatten())}))
                writer.write(example.SerializeToString())

            now_length = mix.shape[-1]
            input_target_length = int(self.duration * self.sample_rate)
                
            audio_stride = int(self.duration * self.sample_rate)

            for i in range(0, now_length - input_target_length+1, audio_stride):
                write_audio(i, i + input_target_length)

              
    def _decode(self, serialized_example):
        
        example = tf.io.parse_single_example(
            serialized_example,
            features={
                "noisy": tf.io.VarLenFeature(tf.float32),
                "clean": tf.io.VarLenFeature(tf.float32)})
        
        noisy =  tf.sparse.to_dense(example["noisy"])

        clean =  tf.sparse.to_dense(example["clean"])

        return noisy, tf.reshape(clean, (4000, 22))