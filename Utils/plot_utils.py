import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_1d_paramSearch_cv_results(params, mean_scores, std_scores, best_param, param_name, score_metric_name, xlogscale=False, show_baseline_score=False, baseline_score=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(params, mean_scores)
    ax.fill_between(params, mean_scores - std_scores, mean_scores + std_scores, alpha=.15)
    if xlogscale==True:
        ax.set_xscale("log")

    ax.set_ylabel(score_metric_name)
    ax.set_xlabel(param_name)
    ax.axvline(best_param, c="C1")
    if show_baseline_score==True:
        ax.axhline(baseline_score, color="grey", linestyle="--")
    ax.grid(True)

def plot_diag_lambda_mat(lambda_mat, feature_names, model_name, savepath=None, vertical_plot=False):
    try: 
        # LGMLVQ
        assert len(feature_names) == lambda_mat.shape[2] == lambda_mat.shape[1]
        lambda_to_plot = []
        for i in range(lambda_mat.shape[0]):
            lambda_to_plot.append(np.diag(lambda_mat[i]))

    except:
        try: 
            # GMLVQ
            assert len(feature_names) == lambda_mat.shape[1] == lambda_mat.shape[0]
            lambda_to_plot = [np.diag(lambda_mat)]
        except: 
            # lambda_mat is an array (only diag elements)
            assert len(feature_names) == len(lambda_mat)
            lambda_to_plot = [lambda_mat]

    feature_dim = len(feature_names)
    num_plots = len(lambda_to_plot)
    fig, ax = plt.subplots(num_plots,1, figsize=(8,3*num_plots))
    plot_title = model_name
    fig.suptitle(plot_title)

    if num_plots > 1:
        colors = ["steelblue", "orange"]
        for i in range(num_plots):
            ax[i].bar(range(feature_dim),lambda_to_plot[i], color=colors[i])
            ax[i].set_xticks(np.arange(feature_dim),feature_names)
            ax[i].set_ylabel("Weight")
            ax[i].grid(True, axis='y')

            plt.setp(ax[i].get_xticklabels(), rotation=30, ha="right",
                    rotation_mode="anchor")
    else:
        if vertical_plot:
            fig,ax = plt.subplots(figsize=(2.3,9))
            ax.set_xlabel("Weight", fontsize=14)
            ax.set_yticklabels([]) # Hide the left y-axis tick-labels
            ax.set_yticks([]) # Hide the left y-axis ticks
            ax.grid(True, axis='x')

            ax1 = ax.twinx() # Create a twin x-axis
            ax1.barh(range(feature_dim), lambda_to_plot[0], align='center') # Plot using `ax1` instead of `ax`
            ax1.set_yticks(range(feature_dim))
            ax1.set_yticklabels(feature_names)
            ax1.invert_yaxis()  # labels read top-to-bottom
            ax.invert_xaxis()
        else:
            ax.bar(range(feature_dim),lambda_to_plot[0])
            ax.set_xticks(np.arange(feature_dim),feature_names)
            ax.set_ylabel("Values")
            ax.grid(True, axis='y')

            plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                    rotation_mode="anchor")
            
    plt.tight_layout()
    if savepath != None:
        fig.savefig(savepath)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_full_relevance_matrix(matrix, labels, savepath=None, showfig=False, cmap='YlGn'):
    # fig, ax = plt.subplots(1,1,figsize=(12,10))
    # im, cbar = heatmap(matrix, labels, labels, ax=ax, cmap=cmap)
    # texts = annotate_heatmap(im, valfmt="{x:.2f}")
    # #fig.suptitle("Relevance Matrix from GMLVQ Model")
    # fig.tight_layout()
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax = sns.heatmap(
        matrix, 
        vmin=-0.3, 
        vmax=0.35, 
        annot=True, 
        fmt='.2f', 
        linewidths=2, 
        cmap='coolwarm',
        ax=ax)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Feature', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=36, ha="right",rotation_mode="anchor")
    #plt.setp(ax.get_yticklabels(), rotation=90, ha="right",rotation_mode="anchor")
    plt.yticks(rotation=0) 
    plt.tight_layout()
    
    if showfig==True:
        plt.show()
    if savepath!=None:
        fig.savefig(savepath)

if __name__ == '__main__':
    print("Running utils.py")