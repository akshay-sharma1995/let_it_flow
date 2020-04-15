import math
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, help="height", required=True)
    parser.add_argument("--w", type=int, help="width", required=True)
    parser.add_argument("--k", type=int, help="filter size (assume h==w)", default=3)
    parser.add_argument("--s",type=int, help="stride range", default=1)
    parser.add_argument("--p",type=int, help="padding range",default=0)
    return parser.parse_args()
    
def main(args):
    args = parse_args()
    dilation = 1
    s = args.s
    p = args.p
    h = args.h
    w = args.w
    filter_size = args.k
    
    possibilites = 0

    for padding in range(p+1):
        for stride in range(1,s+1):
            out_h = (h + 2*padding - filter_size) / (stride) + 1
            out_w = (w + 2*padding - filter_size) / (stride) + 1
            if((out_h - out_h//1 == 0) and (out_w - out_w//1 == 0)):
                print("Input: ({} x {}) --> k: {}, s: {}, p:{} --> output: ({:.0f} x {:.0f})".format(w, h, filter_size, stride, padding, out_w, out_h))
                print(" ")
                possibilites += 1
    if (not possibilites):
        print("No integral output dimension possible with the given configuration")

if __name__=="__main__":
    main(sys.argv)


