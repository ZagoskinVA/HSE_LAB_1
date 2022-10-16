import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle, extract_target
import pandas as pd
import src.config as cfg
from train import train_model
import json

@click.command()
@click.argument('input_train_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('score_filepath', type=click.Path())
def main(input_train_filepath, input_target_filepath, output_filepath, score_filepath):
    train = pd.read_pickle(input_train_filepath)
    target = pd.read_pickle(input_target_filepath)
    score, model = train_model(train, target)
    model.save_model(output_filepath)
    with open(score_filepath, 'w') as fp:
        json.dump(score, fp)

    logger = logging.getLogger(__name__)
    logger.info('train model')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
