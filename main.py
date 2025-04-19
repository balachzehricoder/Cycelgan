import argparse
from train import train
from test import test

def main():
    parser = argparse.ArgumentParser(description='CycleGAN')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    
   
    parser.add_argument('--input_dir', type=str, default='testA')
    parser.add_argument('--output_dir', type=str, default='test_results')
    parser.add_argument('--checkpoint_G', type=str, default='G_A2B_epoch_1.pth')

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test(args.input_dir, args.output_dir, args.checkpoint_G)

if __name__ == "__main__":
    main()
