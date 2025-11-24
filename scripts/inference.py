import argparse
from util import setup_logging, get_logger, load_config_yaml
from models.dlunet import inference as dl_inference_main
from models.random_forest import RandomForestInference

import os


def main():
    parser = argparse.ArgumentParser(description='Run inference for models')
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    parser.add_argument('--model', help='Path to model weights')
    parser.add_argument('--model-type', choices=['dlunet','rf'], default='dlunet')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--log-dir', default='logs', help='Directory to write logs')
    parser.add_argument('--log-format', default='text', choices=['text', 'json'], help='Log output format')
    args = parser.parse_args()

    setup_logging(log_dir=args.log_dir, log_format=args.log_format)
    logger = get_logger(__name__)
    cfg = load_config_yaml(args.config)
    logger.info('Starting inference')

    if args.model_type == 'dlunet':
        dl_inference_main(cfg, args.model, args.output)
    else:
        rf_engine = RandomForestInference(args.model, cfg)
        results = rf_engine.predict_batch(cfg["DLRandomForest"], [f for f in os.listdir(cfg["image_dir"] + "/pictures")], args.output)
        logger.info('Random Forest inference completed')


if __name__ == '__main__':
    main()
