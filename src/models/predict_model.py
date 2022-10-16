import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle, extract_target
import pandas as pd
from train import build_preprocess_pipe
from catboost import CatBoostClassifier
import src.config as cfg

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('input_model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, input_model_filepath, output_filepath):
    df = pd.read_pickle(input_filepath)
    model = CatBoostClassifier().load_model(input_model_filepath)
    predict_pipe = build_preprocess_pipe()
    data = predict_pipe.fit_transform(df)
    predictions = model.predict(data)
    for prediction in predictions:
        for i in range(0, len(cfg.TARGET_COLS)):
            df[cfg.TARGET_COLS[i]] = prediction[i]   
    df.to_csv(output_filepath)
    logger = logging.getLogger(__name__)
    logger.info('predictions')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
