import os

import click
import matplotlib.pyplot as plt
import pymc3 as pm

from lib import get_sample
from lib import load_df
from lib import pairwise_distance_ranks
from lib import pickle_model
from lib import plot_weights
from lib import posterior_predictive_check
from lib import posterior_predictive_pvals
from lib import run_model
from lib import select_k_best_genes


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
@click.option('--num-training-genes', 'n_train', default=50, type=int, show_default=True,
              help='If gene-list is empty, will use SelectKBest to choose gene set.')
def cli(sample, background, name, out_dir, group, col_skip, n_bg, gene_list, n_train):
    # Load input data
    print('Loading input data')
    sample = get_sample(sample, name)
    df = load_df(background)
    df = df.sort_values(group)

    # Parse training genes
    genes = df.columns[col_skip:]
    if gene_list is None:
        print('No gene list provided. Selecting genes via SelectKBest (ANOVA F-value)')
        training_genes = select_k_best_genes(df, genes, group=group, n=n_train)
    else:
        with open(gene_list, 'r') as f:
            training_genes = [x.strip() for x in f.readlines()]

    # Output
    out_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # Select training set
    print('Selecting training set')
    ranks = pairwise_distance_ranks(sample, df, genes, group)
    ranks_out = os.path.join(out_dir, 'ranks.tsv')
    ranks.to_csv(ranks_out, sep='\t')
    n_bg = n_bg if n_bg < len(ranks) else len(ranks)
    train_set = df[df[group].isin(ranks.head(n_bg).Group)]

    # Run Model
    model, trace = run_model(sample, train_set, training_genes, group=group)

    # Traceplot
    fig, axarr = plt.subplots(3, 2, figsize=(10, 5))
    pm.traceplot(trace, varnames=['a', 'b', 'eps'], ax=axarr)
    traceplot_out = os.path.join(out_dir, 'traceplot.png')
    fig.savefig(traceplot_out)

    # Weight plot
    classes = train_set[group].unique()
    weight_out = os.path.join(out_dir, 'weights.png')
    plot_weights(classes, trace, output=weight_out)

    # PPC / PPP
    ppc = posterior_predictive_check(trace, training_genes)
    ppp = posterior_predictive_pvals(sample, ppc)
    ppp_out = os.path.join(out_dir, 'pvals.tsv')
    ppp.to_csv(ppp_out, sep='\t')

    # Save Model
    model_out = os.path.join(out_dir, 'model.pkl')
    pickle_model(model_out, model, trace)
