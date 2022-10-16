import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import click
import logging
from catboost import CatBoostClassifier
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def plot_feature_importance(importance, names, output_filepath):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.savefig(output_filepath)

@click.command()
@click.argument('input_model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_model_filepath, output_filepath):
    model = CatBoostClassifier().load_model(input_model_filepath)
    plot_feature_importance(model.get_feature_importance(), model.feature_names_, output_filepath)
    logger = logging.getLogger(__name__)
    logger.info('visuallize')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
