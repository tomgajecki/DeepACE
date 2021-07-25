# -*- coding: utf-8 -*-
"""
DeepACE
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import argparse

def setup():
    parser = argparse.ArgumentParser(description = 'Main configuration')
    
    parser.add_argument('-top', '--topology', type=str, default = "DeepACE")
    
    parser.add_argument('-mo', '--mode', type=str, default='train')  
    
    parser.add_argument('-gpu',  '--GPU', type=bool, default = True)
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-ld', '--model_dir', type=str, default='./models')    
    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('-md', '--metadata_dir', type=str, default='./data/test/metadata')
    parser.add_argument('-sr', '--sample_rate', type=int, default=8000)
    parser.add_argument('-c',  '--causal', type=bool, default = True)
    parser.add_argument('-me', '--max_epoch', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-k',  '--skip', type=bool, default = True)
    parser.add_argument('-d',  '--duration', type=int, default=4)
     
    parser.add_argument('-N', '-N', type=int, default=128)
    parser.add_argument('-L', '-L', type=int, default=16)
    parser.add_argument('-B', '-B', type=int, default=32)
    parser.add_argument('-H', '-H', type=int, default=64)
    parser.add_argument('-S', '-S', type=int, default=512)
    parser.add_argument('-P', '-P', type=int, default=1024)
    parser.add_argument('-X', '-X', type=int, default=2)
    parser.add_argument('-R', '-R', type=int, default=2)
    parser.add_argument('-G', '-G', type=int, default=16)
     
    args = parser.parse_args()

    return args