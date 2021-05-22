"""Galley of matplotlib or seaborn figures. 
Functions can be run independently.

Dataset explanation can be accessed from here. 
- anes96 : https://www.statsmodels.org/devel/datasets/generated/anes96.html

"""
from typing import Union, Optional, List, Dict, Callable, Any, Tuple
from types import ModuleType

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns

import statsmodels.api as sm

# can install by `pip install contextplt`
import contextplt as cplt

def illustration_of_text_objects():
    x = np.random.rand(100)*100
    y = np.random.rand(100)*100
    with cplt.Single(xlabel="xlabel", ylabel="ylabel", title="title") as p:
        p.ax.scatter(x,y, s=1)
        p.ax.text(10, 10, "Absolute position")
        p.ax.text(0.4,0.4, "Relative position", transform=p.ax.transAxes)
        p.ax.text(0.95,0.03, "Relative position of\n Figure object", 
                    transform=p.fig.transFigure, fontsize=8, ha="right")

def illustration_of_legend_objects():
    with cplt.Single(xlabel="xlabel", ylabel="ylabel", title="title") as p:
        for i in range(3):
            x = np.random.rand(100)*100
            y = np.random.rand(100)*100
            p.ax.scatter(x,y, s=1, label=f"sample{i}")
        plt.legend(loc=(1.00,0.7), frameon=False)

def create_legend_by_oneself():
    patches = create_patch_for_label(label_names = ["test1", "test2", "test3", "NAN"], 
                                     color=["red","blue", "orange", "grey"] , line=True)
    fig = plt.figure(figsize=(4,4), dpi=200)
    ax = fig.add_subplot(111)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.legend(handles=patches, frameon=False)
    plt.show()

def create_patch_for_label(
    label_names: List[str], 
    label_title: str = "", 
    cmap_name: str = "tab10", 
    color : Union[List[str], List[Tuple]] = None,
    line : bool = False,
    ) -> List[mpatches.Patch]:
    """Create list of patches for legend.

    Args:
        label_names : list of label names. 
        label_title : title of label handle.
        cmap_name : colormap name. 
        color : If color is specified, use this color set to display.
        line : legend becomes line style. 

    Examples:
        >>> patches = gallery.create_patch_for_label(label_names = ["test1", "test2", "test3"], color=["red","blue", "orange"] , line=True)
        >>> fig = plt.figure(figsize=(6,6), dpi=300 )
        >>> ax = fig.add_subplot(111)
        >>> ax.axes.xaxis.set_visible(False)
        >>> ax.axes.yaxis.set_visible(False)
        >>> plt.legend(handles=patches, frameon=False)
        >>> plt.show()
    """
    cmap = plt.get_cmap(cmap_name)
    patches = []
    for i, name in enumerate(label_names):
        if name == "NAN":
            c = "grey"
        elif color == None:
            c = cmap(i)
        else:
            c = color[i]
            
        if line:
            patch = Line2D([0], [0], color=c, label=name)
        else:
            patch = mpatches.Patch(color=c, label=name)
        patches.append(patch)
    return(patches)

def run_simple_scatter():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    simple_scatter(df, x, y)

def simple_scatter(df : pd.DataFrame, x : str, y : str) -> None:
    with cplt.Single(xlabel=x, ylabel=y, title="scatterplot") as p:
        p.ax.scatter(df[x], df[y], s=1)

def run_scatter_with_linear_reg():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    scatter_with_linear_reg(df, x, y)

def scatter_with_linear_reg(df : pd.DataFrame, x : str, y : str) -> None:
    with cplt.Single(xlabel=f"log[{x}]", ylabel=y, 
                           title="scatter with linear regression", xlim=[10, 95]) as p:
        sns.regplot(data=df, x=x, y=y, ax=p.ax, 
                    scatter_kws=dict(s=1, color="purple"), 
                    line_kws=dict(color="green"))

def run_contourplot():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    contourplot(df, x, y)

def contourplot(df : pd.DataFrame, x : str, y : str) -> None:
    with cplt.Single(figsize=(6,5), title=f"contour plot. {x} and {y}") as p:
        sns.kdeplot(data=df, x=x, y =y, 
                    common_norm=False, fill=True, ax=p.ax, n_levels=10, 
                    cbar=True, thresh=0, cmap='viridis' )

def run_histogram2d():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    bins = (20,20)
    rng= ((10,100), (-3, 10))
    histogram2d(df, x, y, bins, rng)


def histogram2d(df: pd.DataFrame, x : str, y : str, 
                bins : Tuple[int,int], rng : Tuple[Tuple[int,int,], Tuple[int,int]]
                ) -> None:
    with cplt.Single(xlabel=x, ylabel=y, title="2D histogram",
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
    with cplt.Single(xlabel=x, ylabel=y, title="stratified scatter plot") as p:
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

def run_stacked_histogram():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"
    stacked_histogram(df, x, y, c)

def stacked_histogram(df : pd.DataFrame, x : str, y : str, c : str) -> None:
    color_n = len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with cplt.Single() as p:
        sns.histplot(data=df,x=x,hue=c, fill=True, palette=palette , alpha=1 )

def run_kde_density_with_stratification():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"
    kde_density_with_stratification(df, x, y, c)

def kde_density_with_stratification(
        df : pd.DataFrame, x : str, y : str, c : str
    ) -> None:
    color_n =  len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with cplt.Single() as p:
        sns.kdeplot(data=df, x=x, hue=c, ax=p.ax, 
                    common_norm=False, fill=True, alpha=0.3, bw_adjust=0.5, 
                    palette=palette)
    
def run_kde_density_area_plot():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"
    kde_density_area_plot(df, x, y, c)

def kde_density_area_plot(
        df : pd.DataFrame, x : str, y : str, c : str
    ) -> None:
    color_n =  len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with cplt.Single() as p:
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

def run_stacked_hist_kde_density_and_area_plot_with_stratification():
    anes96 = sm.datasets.anes96
    df = anes96.load_pandas().data
    x = "age"
    y = "logpopul"
    c = "educ"
    stacked_hist_kde_density_and_area_plot_with_stratification(df, x, y , c)

def stacked_hist_kde_density_and_area_plot_with_stratification(
        df : pd.DataFrame, x : str, y : str, c : str,
    ) -> None:
    color_n =  len(df[c].unique())
    palette = list(plt.cm.tab10.colors[:color_n])
    with cplt.Multiple(figsize=(6,8), dpi=150,grid=(3,1), label_outer=True,
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


