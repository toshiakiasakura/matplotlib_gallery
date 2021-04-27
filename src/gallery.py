"""Galley of matplotlib or seaborn figures. 
Functions can be run independently.

Dataset explanation can be accessed from here. 
- anes96 : https://www.statsmodels.org/devel/datasets/generated/anes96.html

"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

# can install by `pip install contextplt`
import contextplt

def simple_scatter():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    with contextplt.Single(xlabel=x, ylabel=y, title="scatterplot") as p:
        p.ax.scatter(df[x], df[y], s=1)

def scatter_with_linear_reg():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    with contextplt.Single(xlabel=f"log[{x}]", ylabel=y, 
                           title="scatter with linear regression", xlim=[10, 95]) as p:
        sns.regplot(data=df, x=x, y=y, ax=p.ax, 
                    scatter_kws=dict(s=1, color="purple"), 
                    line_kws=dict(color="green"))

def contourplot():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    with contextplt.Single(figsize=(6,5), title=f"contour plot. {x} and {y}") as p:
        sns.kdeplot(data=df, x=x, y =y, 
                    common_norm=False, fill=True, ax=p.ax, n_levels=10, 
                    cbar=True, thresh=0, cmap='viridis' )

def histogram2d():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"

    bins = (20,20)
    rng= ((10,100), (-3, 10))
    with contextplt.Single(xlabel=x, ylabel=y, title="2D histogram",
                           figsize=(7,5)) as p:
        H =  p.ax.hist2d(df[x], df[y], bins=bins, cmap=plt.cm.jet, 
                         density=True, cmin=0, cmax=None, range=rng)
        p.fig.colorbar(H[3],ax=p.ax)

def run_stratified_scatter():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"
    with contextplt.Single(xlabel=x, ylabel=y, title="stratified scatter plot") as p:
        stratified_scatter(p.ax, df, x, y, c)

def stratified_scatter(ax, df_: pd.DataFrame, x: str, y: str, c: str) -> None:
    """
    
    Args:
        ax : axis object.
        df_ : dataframe to be plotted.
        x : a column name for x axis.
        y : a column name for y axis.
        c : a column name for stratification.
    """
    
    columns = sorted(df_[c].unique())
    for col in columns:
        cond = df_[c] == col
        dfM = df_.loc[cond]
        ax.scatter(dfM[x], dfM[y], s=1, label=str(col))
        plt.legend(bbox_to_anchor=(1, 0.98), frameon = False)

def stacked_histgram():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"

    color_n = len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with contextplt.Single() as p:
        sns.histplot(data=df,x=x,hue=c, fill=True, palette=palette , alpha=1 )

def kde_density_with_stratification():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"

    color_n =  len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with contextplt.Single() as p:
        sns.kdeplot(data=df, x=x, hue=c, ax=p.ax, 
                    common_norm=False, fill=True, alpha=0.3, bw_adjust=0.5, 
                    palette=palette)
    

def kde_density_area_plot():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"

    color_n =  len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with contextplt.Single() as p:
        sns.kdeplot(data=df, x=x, hue=c, ax=p.ax, 
                    common_norm=True, multiple="fill", fill=True, 
                    bw_adjust=0.5, palette=palette, alpha=1, linewidth=0.1 )
        move_legend(p.ax, bbox_to_anchor=(1,0.98))

def move_legend(ax, new_loc="upper left", **kws):
    """move legend created by seaborn. See issues in seaborn. 
    https://github.com/mwaskom/seaborn/issues/2280
    """
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

def stacked_hist_kde_density_and_area_plot_with_stratification():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"

    color_n =  len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with contextplt.Multiple(figsize=(6,8), dpi=150,grid=(3,1), label_outer=True,
                             suptitle="stacked hist., kde density and area plot", 
                             ) as p:
        ax = p.set_ax(1)
        sns.histplot(data=df,x=x,hue=c, fill=True, palette=palette , alpha=1 )
        move_legend(ax, bbox_to_anchor=(1,0.98))

        ax = p.set_ax(2)
        sns.kdeplot(data=df, x=x, hue=c, ax=ax, 
                    common_norm=False, fill=True, alpha=0.3, bw_adjust=0.5, 
                    palette=palette )
        move_legend(ax, bbox_to_anchor=(1,0.98))

        ax = p.set_ax(3)
        sns.kdeplot(data=df, x=x, hue=c, ax=ax, 
                    common_norm=True, multiple="fill", fill=True, 
                    bw_adjust=0.5, palette=palette, alpha=1, linewidth=0.1 )
        move_legend(ax, bbox_to_anchor=(1,0.98))


