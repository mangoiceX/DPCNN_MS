
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="mutable parameters")
    parser.add_argument('-pre', '--use_pre_embed', type=int, default=0)

    args = vars(parser.parse_args())

    return args

args = parse_args()