#!/usr/bin/env python3

import argparse


def set_args():
    parser = argparse.ArgumentParser("--NER")
    parser.add_argument("--train_data_path", default="./data/train.json", type=str)
    parser.add_argument("--valid_data_path", default="./data/dev.json", type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_seq_len", default=64, type=int)
    return parser.parse_args()
