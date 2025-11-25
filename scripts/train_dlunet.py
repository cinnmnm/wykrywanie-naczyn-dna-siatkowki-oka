import argparse
from util import setup_logging, get_logger, load_config_yaml
from models.dlunet import train as train_main

import logging


def main():
    parser = argparse.ArgumentParser(description='Train DL Unet model')
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    parser.add_argument('--log-dir', default='logs', help='Directory to write logs')
    parser.add_argument('--log-format', default='text', choices=['text', 'json'], help='Log output format')
    args = parser.parse_args()

    setup_logging(log_dir=args.log_dir, log_format=args.log_format)
    logger = get_logger(__name__)
    cfg = load_config_yaml(args.config)
    logger.info('Starting DL Unet training')
    train_main(cfg)


if __name__ == '__main__':
    main()
