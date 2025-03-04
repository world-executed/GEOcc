import argparse
import mmcv
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataloader, build_dataset
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pkl', help='test pkl file path') 
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl)
    print(dataset.evaluate(outputs))

if __name__ == '__main__':
    main()