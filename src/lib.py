import pickle
from typing import List, Dict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import pairwise_distances
from tqdm.autonotebook import tqdm


def select_k_best_genes(df: pd.DataFrame, genes: List[str], group='tissue', n=50) -> List[str]:
    """
    Select K genes based on ANOVA F-value

    Args:
        df: Background dataset
        genes: Genes to include in selection process
        group: Column in dataset that distinguishes groups
        n: Number of genes to select (k)

    Returns:
        List of selected genes
    """
    k = SelectKBest(k=n)
    k.fit_transform(df[genes], df[group])
    return [genes[i] for i in k.get_support(indices=True)]


def get_sample(df_path: str, sample_name: str) -> pd.Series:
    """
    Loads dataframe containing sample and returns sample

    Args:
        df_path: Path to DataFrame containing sample
        sample_name: Name of sample in the index of the DataFrame

    Returns:
        Sample vector
    """
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path, index_col=0)
    elif df_path.endswith('.tsv'):
        df = pd.read_csv(df_path, sep='\t', index_col=0)
    else:
        try:
            df = pd.read_hdf(df_path)
        except:
            raise RuntimeError(f"Failed to open DataFrame: {df_path}")

    if sample_name in df.index:
        return df.loc[sample_name]
    else:
        raise RuntimeError(f"Sample {sample_name} not located in index of DataFrame {df_path}")


def load_df(df_path: str) -> pd.DataFrame:
    """
    Load background DataFrame

    Args:
        df_path: Path to DataFrame

    Returns:
        Background DataFrame
    """
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path, index_col=0)
    elif df_path.endswith('.tsv'):
        df = pd.read_csv(df_path, sep='\t', index_col=0)
    else:
        try:
            df = pd.read_hdf(df_path)
        except:
            raise RuntimeError(f"Failed to open DataFrame: {df_path}")
    return df


def pairwise_distance_ranks(sample: pd.Series, df: pd.DataFrame, genes: List[str], group: str) -> pd.DataFrame:
    """
    Calculate pairwise distance, rank, and take group median

    Args:
        sample: n-of-1 sample. Gets own label
        df: background dataset
        genes: genes to use for pairwise distance
        group: Column to use as class discriminator

    Returns:
        DataFrame of pairwise distance ranks
    """
    dist = pairwise_distances(np.array(sample[genes]).reshape(1, -1), df[genes])
    dist = pd.DataFrame([dist.ravel(), df[group]]).T
    dist.columns = ['Distance', 'Group']
    dist = dist.sort_values('Distance')

    # Pandas-FU
    dist = dist.reset_index(drop=True).reset_index()
    return dist.groupby('Group').apply(lambda x: x['index'].median()).sort_values().reset_index(name='MedianRank')


def run_model(sample: pd.Series,
              df: pd.DataFrame,
              training_genes: List[str],
              group: str = 'tissue',
              **kwargs):
    """
    Run Bayesian model using prefit Y's for each Gene and Dataset distribution

    Args:
        sample: N-of-1 sample to run
        df: Background dataframe to use in comparison
        training_genes: Genes to use during training
        group:
        **kwargs:

    Returns:
        Model and Trace from PyMC3
    """
    # Importing here since Theano base_compiledir needs to be set prior to import
    import pymc3 as pm

    classes = sorted(df[group].unique())
    df = df[[group] + training_genes]

    # Collect fits
    ys = {}
    for gene in training_genes:
        for i, dataset in enumerate(classes):
            cat_mu, cat_sd = st.norm.fit(df[df[group] == dataset][gene])
            # Standard deviation can't be initialized to 0, so set to 0.1
            cat_sd = 0.1 if cat_sd == 0 else cat_sd
            ys[f'{gene}-{dataset}'] = (cat_mu, cat_sd)

    click.echo('Building model')
    with pm.Model() as model:
        # Linear model priors
        a = pm.Normal('a', mu=0, sd=1)
        b = [1] if len(classes) == 1 else pm.Dirichlet('b', a=np.ones(len(classes)))
        # Model error
        eps = pm.InverseGamma('eps', 2.1, 1)

        # Linear model declaration
        for gene in tqdm(training_genes):
            mu = a
            for i, dataset in enumerate(classes):
                name = f'{gene}-{dataset}'
                y = pm.Normal(name, *ys[name])
                mu += b[i] * y

            # Embed mu in laplacian distribution
            pm.Laplace(gene, mu=mu, b=eps, observed=sample[gene])
        # Sample
        trace = pm.sample(**kwargs)
    return model, trace


def calculate_weights(groups: List[str], trace) -> pd.DataFrame:
    """
    Calculates weights of the background groups by examining the beta coefficient in the trace

    Args:
        groups: List of groups trained in the model
        trace: PyMC3 Trace

    Returns:
        DataFrame of classes and weights
    """
    class_col = []
    for c in groups:
        class_col.extend([c for _ in range(len(trace['a']))])

    weight_by_class = pd.DataFrame({'Class': class_col,
                                    'Weights': np.array([trace['b'][:, x] for x in range(len(groups))]).ravel()})
    return weight_by_class


def plot_weights(groups: List[str], trace, output: str = None):
    """
    Plot model coefficients associated with each group

    Args:
        groups: List of groups trained in the model
        trace: PyMC3 Trace
        output: Optional output location for plot
    """
    # Construct weight by class DataFrame
    weights = calculate_weights(groups, trace)

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=weights, x='Class', y='Weights')
    plt.xticks(rotation=90)
    plt.title('Median Beta Coefficient Weight by Tissue for N-of-1 Sample')
    if output:
        plt.savefig(output, bbox_inches='tight')
    return weights


def posterior_predictive_check(trace, genes: List[str]) -> Dict[str, np.array]:
    """
    Posterior predictive check for a list of genes trained in the model

    Args:
        trace: PyMC3 trace
        genes: List of genes of interest

    Returns:
        Dictionary of [genes, array of posterior sampling]
    """
    d = {}
    for gene in genes:
        d[gene] = _gene_ppc(trace, gene)
    return d


def _gene_ppc(trace, gene: str) -> np.array:
    """
    Calculate posterior predictive for a gene

    Args:
        trace: PyMC3 Trace
        gene: Gene of interest

    Returns:
        Random variates representing PPC of the gene
    """
    y_gene = [x for x in trace.varnames if x.startswith(f'{gene}-')]
    b = trace['a']
    for i, y_name in enumerate(y_gene):
        b += trace['b'][:, i] * trace[y_name]
    return np.random.laplace(loc=b, scale=trace['eps'])


def posterior_predictive_pvals(sample: pd.Series, ppc: Dict[str, np.array]) -> pd.Series:
    """
    Produces Series of posterior p-values for all genes in the Posterior Predictive Check (PPC) dictionary

    Args:
        sample: N-of-1 sample
        ppc: Posterior predictive check dictionary

    Returns:
        Genes and their posterior p-value
    """
    pvals = {}
    for gene in ppc:
        z_true = sample[gene]
        z = ppc[gene]
        pvals[gene] = _ppp_one_gene(z_true, z)
    return pd.Series(pvals).sort_values()


def _ppp_one_gene(z_true, z):
    """Calculates ppp for one gene"""
    return round(np.sum(z_true < z) / len(z), 5)


def pickle_model(output_path: str, model, trace):
    """Pickles PyMC3 model and trace"""
    with open(output_path, 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace}, buff)
