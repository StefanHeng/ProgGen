"""
Embed sentences w/ SBert, plot them in 2D, 3D space

Compare generated data w/ test set
"""

from os.path import join as os_join
from typing import List, Tuple, Dict, Iterable, Union

import pandas as pd

from stefutil import get_logger, ca
from src.util import save_fig
from src.data_util.stats import NerDatasetStats


_logger = get_logger('Viz')


# def confidence_ellipse(ax_, x, y, n_std=1., **kws):
#     """
#     Modified from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
#     Create a plot of the covariance confidence ellipse of x and y
#
#     :param ax_: matplotlib axes object to plot ellipse on
#     :param x: x values
#     :param y: y values
#     :param n_std: number of standard deviations to determine the ellipse's radius'
#     :return matplotlib.patches.Ellipse
#     """
#     cov = np.cov(x, y)
#     pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
#     r_x, r_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
#     _args = {**dict(fc='none'), **kws}
#     ellipse = Ellipse((0, 0), width=r_x * 2, height=r_y * 2, **_args)
#     scl_x, scl_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
#     mu_x, mu_y = np.mean(x), np.mean(y)
#     tsf = transforms.Affine2D().rotate_deg(45).scale(scl_x, scl_y).translate(mu_x, mu_y)
#     ellipse.set_transform(tsf + ax_.transData)
#     return ax_.add_patch(ellipse)


class Visualizer:
    def __init__(
            self, dataset_name: str = 'conll2003-no-misc', dataset_dir_names: List[str] = None, plot_names: List[str] = None,
            color_palette: str = 'husl', postfix: str = None, plot_type: str = 'sentence', weigh_entities: bool = False
    ):
        """
        :param dataset_name: Dataset name, e.g. 'conll2003'
        :param dataset_dir_names: List of generated data directory names to compare against
        :param plot_names: List of names corresponding to `dataset_dir_names` for plotting
        :param color_palette: Color palette for plotting
        :param postfix: Postfix description for plot
        :param plot_type: Type of plot to generate, one of [`sentence`, `entity`]
        :param weigh_entities: Whether to weigh entity vectors by their frequency
        """
        self.dataset_name = dataset_name
        self.dataset_dir_names = dataset_dir_names
        if plot_names is None:
            self.plot_names = dataset_dir_names
        else:
            assert len(plot_names) == len(dataset_dir_names)  # sanity check
            self.plot_names = plot_names

        # use one shared test stats object
        self.weigh_entities = weigh_entities
        self.stats: Dict[str, NerDatasetStats] = {dnm: self._load_single(dir_name=dnm) for dnm in dataset_dir_names}
        self._train_stats, self.test_stats = None, self.stats[dataset_dir_names[0]].test_stats
        for dnm, stats in self.stats.items():
            stats.test_stats = self.test_stats

        self.color_palette = color_palette
        self.postfix = postfix
        ca(plot_type=plot_type)
        self.plot_type = plot_type

        self.k_setup = 'setup'  # syntax sugar for df and plotting
        self.setup_train, self.setup_test = 'train set', 'test set'

        self.split2setup = {'train': self.setup_train, 'test': self.setup_test}
        self.stats_map = {self.setup_train: self.train_stats, self.setup_test: self.test_stats}

    def _load_single(self, dir_name: str = None) -> NerDatasetStats:
        if self.dataset_name == 'conll2003-no-misc' and dir_name == 'train':
            dir_name = 'train-1k'
        return NerDatasetStats.from_dir_name(
            dataset_name=self.dataset_name, dir_name=dir_name, unique_entity_vector=not self.weigh_entities)

    @property
    def train_stats(self) -> NerDatasetStats:
        if self._train_stats is None:
            self._train_stats = self._load_single(dir_name='train')
            self._train_stats.test_stats = self.test_stats
        return self._train_stats

    def _iter_stats(self, include_ori: Union[bool, List[str]] = False) -> Iterable[Tuple[str, NerDatasetStats]]:
        """
        Iterate over stats objects
        """
        dirs_ori = []
        if include_ori is True:
            dirs_ori = [self.setup_train, self.setup_test]
        elif isinstance(include_ori, list):
            assert all(dnm in ['train', 'test'] for dnm in include_ori)
            dirs_ori = [self.split2setup[dnm] for dnm in include_ori]

        if len(dirs_ori) > 0:
            for dnm in dirs_ori:
                yield dnm, self.stats_map[dnm]
        for nm, stat in zip(self.plot_names, self.stats.values()):
            yield nm, stat

    def plot_embedding_similarity(self, save: bool = True):
        """
        For each generated dataset, plot histogram of max embedding cosine similarity
            See `src/visualize/stats::NerDatasetStats::test_vector_overlap`
        """
        import matplotlib.pyplot as plt  # lazy import to save time
        import seaborn as sns

        k_sim = 'cosine similarity'
        dfs, cols = [], [k_sim, self.k_setup]
        for nm, stat in self._iter_stats(include_ori=['train']):
            sic(nm)
            sims = stat.test_vector_overlap(kind=self.plot_type).cosine_sims
            dfs.append(pd.DataFrame({k_sim: sims, self.k_setup: nm}))
        df = pd.concat(dfs, ignore_index=True)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        args = dict(palette=self.color_palette, hue=self.k_setup, common_norm=False, stat='percent')  # fill=False,

        def plot_single(cumulative: bool = None, ax: plt.Axes = None):
            args_ = args.copy()
            if cumulative:
                args_.update(dict(cumulative=cumulative, element='step', fill=False))
            else:
                args_['kde'] = True
            sns.histplot(data=df, x=k_sim, **args_, ax=ax)
            ax.set_title('Cumulative' if cumulative else 'Standard')
        plot_single(cumulative=False, ax=axes[0])
        plot_single(cumulative=True, ax=axes[1])
        plt.gca().get_legend().set_title('Data Gen Setups')

        title = f'Distribution of max Cosine Similarity w/ Test {self.plot_type}'
        if self.plot_type == 'entity' and self.weigh_entities:
            title = f'{title}, weighed'
        if self.postfix:
            title = f'{title} ({self.postfix})'
        plt.suptitle(title)
        save_fig(title, save=save)
    
    def plot_embeddings(self, title: str = None, save: bool = True):
        """
        Visualize embeddings in 2D, 3D space
        """
        from stefutil import vector_projection_plot

        nm2vects = dict()
        for nm, stat in self._iter_stats(include_ori=True):
            nm2vects[nm] = stat.sentence_vectors if self.plot_type == 'sentence' else stat.entity_vectors
        setups = list(nm2vects.keys())
        # sanity check
        # assert all(s in [self.setup_train, self.setup_test, 'SimPrompt', 'AttrPrompt on X', 'AttrPrompt on Y'] for s in setups)

        title_ = f'{self.plot_type.capitalize()} Embedding Projections w/ t-SNE'
        if self.plot_type == 'entity' and self.weigh_entities:
            title_ = f'{title_}, weighed'
        if self.postfix:
            title_ = f'{title_} ({self.postfix})'
        vector_projection_plot(name2vectors=nm2vects, key_name=self.k_setup, ellipse_std=1.25, title=None if title == 'none' else title_)
        save_fig(title_, save=save)


if __name__ == '__main__':
    from stefutil import sic

    def check_plot():
        dnm = 'conll2003-no-misc'
        tp = 'sentence'
        # tp = 'entity'
        sic(dnm, tp)
        et_wt = None
        if tp == 'entity':
            # et_wt = False
            et_wt = True
        sic(et_wt)

        # post = 'n=1K'
        # post = 'n=3K'
        # post = 'ablation'
        post = None
        sic(post)
        if post == 'ablation':
            dnms = [
                '23-11-28_NER-Dataset_{fmt=n-p2,#l=50,#a=5}_ori-re-annotate_fix-edge',
                '23-11-28_NER-Dataset_{fmt=n-p2,#l=50,#a=5}_rephrase-re-annotate_fix-edge'
            ]
            dnms = [os_join('step-wise', dnm) for dnm in dnms]
            p_nms = ['Original X + LLM Y', 'Original X rephrased + LLM Y']
        elif post is None:
            dnms = [
                '23-11-08_NER-Dataset_{fmt=n-p2,#l=50}_ppt-v4',
                '23-11-19_NER-Dataset_{fmt=n-p2,#l=3,dc=T}',
                '23-11-19_NER-Dataset_{fmt=n-p2,#l=3,de=T}',
                '23-11-19_NER-Dataset_{fmt=n-p2,#l=3,de=s}',
                '23-11-19_NER-Dataset_{fmt=n-p2,#l=3,ap={dc=T,de=s}}'
            ]
            p_nms = ['Simple Prompt', 'Diversify X', 'Diversify Y (vanilla)', 'Diversify Y (latent)', 'Diversify X + Y']
        else:
            if post == 'n=1K':
                dnms = [
                    '23-11-02_Processed-NER-Data_{fmt=n-p2,#l=50}',
                    '23-11-06_Processed-NER-Data_{fmt=n-p2,#l=3,dc=T}_attr-at-end',
                    '23-11-05_Processed-NER-Data_{fmt=n-p2,#l=3,de=T}_attr-prompt-at-end'
                ]
            elif post == 'n=3K':
                dnms = [
                    '23-11-06_Processed-NER-Data_{fmt=n-p2,#l=50}_n=3K',
                    '23-11-06_Processed-NER-Data_{fmt=n-p2,#l=3,dc=T}_n=3K',
                    '23-11-06_Processed-NER-Data_{fmt=n-p2,#l=3,de=T}_n=3K'
                ]
            else:
                raise NotImplementedError
            p_nms = ['SimPrompt', 'AttrPrompt on X', 'AttrPrompt on Y']
        sic(dnms, p_nms)
        viz = Visualizer(dataset_name=dnm, dataset_dir_names=dnms, plot_names=p_nms, postfix=post, plot_type=tp, weigh_entities=et_wt)
        # viz.plot_embedding_similarity()
        viz.plot_embeddings(title='none')
    check_plot()
