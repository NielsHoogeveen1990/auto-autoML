from auto_autoML.auto_pipeline import AutoML
import click
import click_pathlib
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

space = {

    'est__strategy': {"search": "choice",
                      "space": ["LightGBM"]},
    'est__n_estimators': {"search": "choice",
                          "space": [150]},
    'est__colsample_bytree': {"search": "uniform",
                              "space": [0.8, 0.95]},
    'est__subsample': {"search": "uniform",
                       "space": [0.8, 0.95]},
    'est__max_depth': {"search": "choice",
                       "space": [5, 6, 7, 8, 9]},
    'est__learning_rate': {"search": "choice",
                           "space": [0.07]}

}

@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
    pass


@main.command()
@click.option("--num-drift", type=float, default=0.6)
@click.option("--max-cardinality", type=float, default=0.9)
@click.option("--pred-type", type=str, default='classification')
@click.option("--scoring", type=str, default='accuracy')
@click.option("--n-folds", type=int, default=5)
@click.option("--pop-size", type=int, default=50)
@click.option("--data-path", type=click_pathlib.Path(exists=True), required=True)
@click.option("--target", type=str, required=True)
@click.option("--split-ratio", type=float, default=0.7)
@click.option("--stratified", type=bool, default=True)
@click.option("--max-evals", type=int, default=40)
@click.option("--hyper-params-space", type=dict, required=True, default=space)
def automl(num_drift, max_cardinality, pred_type, scoring, n_folds, pop_size, data_path, target, split_ratio, stratified, max_evals, hyper_params_space):
    auto_ml = AutoML(num_drift, max_cardinality, pred_type, scoring, n_folds, pop_size)
    auto_ml.optimise(data_path, target, split_ratio, stratified, max_evals, hyper_params_space)

    logger.info('Finished with optimising autoML.')
