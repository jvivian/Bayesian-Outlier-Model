import os
import shutil
import time
import warnings

import click
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy import stats as st
from tqdm.autonotebook import tqdm

from outlier_model.lib import get_sample
from outlier_model.lib import load_df
from outlier_model.lib import pairwise_distance_ranks
from outlier_model.lib import pickle_model
from outlier_model.lib import plot_weights
from outlier_model.lib import posterior_predictive_check
from outlier_model.lib import posterior_predictive_pvals
from outlier_model.lib import select_k_best_genes

warnings.filterwarnings('ignore')


def run_model(sample,
              df,
              training_genes,
              weights,
              group: str = 'tissue',
              **kwargs):
    """
    Run Bayesian model using prefit Y's for each Gene and Dataset distribution

    Args:
        sample: N-of-1 sample to run
        df: Background dataframe to use in comparison
        training_genes: Genes to use during training
        group: Column to use to distinguish different groups
        **kwargs:

    Returns:
        Model and Trace from PyMC3
    """
    classes = sorted(df[group].unique())
    df = df[[group] + training_genes]

    # Collect fits
    ys = {}
    for gene in training_genes:
        for i, dataset in enumerate(classes):
            cat_mu, cat_sd = st.norm.fit(df[df[group] == dataset][gene])
            # Standard deviation can't be initialized to 0, so set to 0.1
            cat_sd = 0.1 if cat_sd == 0 else cat_sd
            ys[f'{gene}={dataset}'] = (cat_mu, cat_sd)

    print('Building model')
    with pm.Model() as model:
        # Linear model priors
        a = pm.Normal('a', mu=0, sd=1)
        # Model error
        eps = pm.InverseGamma('eps', 2.1, 1)

        # TODO: Try tt.stack to declare mu more intelligently via b * y
        # Linear model declaration
        for gene in tqdm(training_genes):
            mu = a
            for i, dataset in enumerate(classes):
                name = f'{gene}={dataset}'
                y = pm.Normal(name, *ys[name])
                mu += weights[i] * y

            # Embed mu in laplacian distribution
            pm.Laplace(gene, mu=mu, b=eps, observed=sample[gene])
        # Sample
        trace = pm.sample(**kwargs)
    return model, trace


def weights_from_ranks(ranks, n=5):
    weights = ranks.head(n)
    weights = weights.MedianRank.sum() / weights.MedianRank
    return weights / weights.sum()


@click.command()
@click.option('--sample', required=True, type=str, help='Sample(s) by Genes matrix (csv/tsv/hd5)')
@click.option('--background', required=True, type=str,
              help='Samples by Genes matrix with metadata columns first '
                   '(including a group column that discriminates samples by some category) (csv/tsv/hd5)')
@click.option('--name', required=True, type=str, help='Name of sample in sample matrix')
@click.option('--gene-list', type=str, help='Single column file of genes to use for training')
@click.option('--out-dir', default='.', type=str, help='Output directory')
@click.option('--group', default='tissue', show_default=True, type=str,
              help='Categorical column vector in the background matrix')
@click.option('--col-skip', default=1, show_default=True, type=int,
              help='Number of metadata columns to skip in background matrix so remainder are genes')
@click.option('--num-backgrounds', 'n_bg', default=5, type=int, show_default=True,
              help='Number of background categorical groups to include in the model training')
@click.option('--max-genes', default=100, type=int, show_default=True,
              help='Maximum number of genes to run. I.e. if a gene list is input, how many additional'
                   'genes to add via SelectKBest. Useful for improving beta coefficients'
                   'if gene list does not contain enough tissue-specific genes.')
@click.option('--num-training-genes', 'n_train', default=50, type=int, show_default=True,
              help='If gene-list is empty, will use SelectKBest to choose gene set.')
def cli(sample, background, name, out_dir, group, col_skip, n_bg, gene_list, max_genes, n_train):
    click.clear()
    click.secho('Bayesian Gene Expression Outlier Model', fg='green', bg='black', bold=True)

    # Output
    out_dir = os.path.abspath(os.path.join(out_dir, name))
    theano_dir = os.path.join(out_dir, '.theano')
    os.makedirs(theano_dir, exist_ok=True)

    # Load input data
    click.echo('Loading input data')
    sample = get_sample(sample, name)
    df = load_df(background)
    df = df.sort_values(group)
    genes = df.columns[col_skip:]

    # Select training set
    click.echo('Selecting training set')
    ranks = pairwise_distance_ranks(sample, df, genes, group)
    ranks_out = os.path.join(out_dir, 'ranks.tsv')
    ranks.to_csv(ranks_out, sep='\t')
    n_bg = n_bg if n_bg < len(ranks) else len(ranks)
    train_set = df[df[group].isin(ranks.head(n_bg)['Group'])]

    # Parse training genes
    if gene_list is None:
        click.secho(f'No gene list provided. Selecting {n_train} genes via SelectKBest (ANOVA F-value)', fg='yellow')
        training_genes = select_k_best_genes(train_set, genes, group=group, n=n_train)
    else:
        with open(gene_list, 'r') as f:
            training_genes = [x.strip() for x in f.readlines() if not x.isspace()]
        # Pad training genes with additional genes from SelectKBest based on `max-genes` argument
        if len(training_genes) < max_genes:
            diff = max_genes - len(training_genes)
            click.secho(f'Adding {diff} genes via SelectKBest (ANOVA F-value) to reach {max_genes} total genes',
                        fg='yellow')
            training_genes += select_k_best_genes(train_set, genes, group=group, n=diff)
            training_genes = sorted(set(training_genes))

    # Set env variable for base_compiledir before importing model
    os.environ['THEANO_FLAGS'] = f'base_compiledir={theano_dir}'

    # Calculate weights for model
    weights = weights_from_ranks(ranks)

    # Run model and output runtime
    t0 = time.time()
    model, trace = run_model(sample, train_set, training_genes, weights=weights, group=group)
    runtime = round((time.time() - t0) / 60, 2)
    unit = 'min' if runtime < 60 else 'hr'
    runtime = runtime if runtime < 60 else round(runtime / 60, 2)
    click.secho(f'Model runtime: {runtime} ({unit})', fg='green')

    # Traceplot
    fig, axarr = plt.subplots(3, 2, figsize=(10, 5))
    pm.traceplot(trace, varnames=['a', 'eps'], ax=axarr)
    traceplot_out = os.path.join(out_dir, 'traceplot.png')
    fig.savefig(traceplot_out)

    # Weight plot and weight table
    classes = train_set[group].unique()
    weights = plot_weights(classes, trace, output=os.path.join(out_dir, 'weights.png'))
    # Convert weights to summarized information of median and std
    weights = weights.groupby('Class').agg({'Weights': ['median', 'std']})
    weights = weights.sort_values(('Weights', 'median'), ascending=False)
    weights.to_csv(os.path.join(out_dir, 'weights.tsv'), sep='\t')

    # PPC / PPP
    ppc = posterior_predictive_check(trace, training_genes)
    ppp = posterior_predictive_pvals(sample, ppc)
    ppp_out = os.path.join(out_dir, 'pvals.tsv')
    ppp.to_csv(ppp_out, sep='\t')

    # Save Model
    model_out = os.path.join(out_dir, 'model.pkl')
    pickle_model(model_out, model, trace)

    # Cleanup
    shutil.rmtree(theano_dir)
