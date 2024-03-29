import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

plt.rcParams["savefig.facecolor"] = "w"
sns.set_style("whitegrid")


def plot_rate_dist(X, Y, path=""):
    "plot rate distributions for two populations"

    fig, ax = plt.subplots()

    r1 = np.mean(X, axis=0)
    r2 = np.mean(Y, axis=0)

    sns.histplot(
        ax=ax,
        data={f"X (n = {len(r1)})": r1, f"Y (n = {len(r2)})": r2},
        binwidth=1,
        stat="density",
        common_norm=False,
    )
    ax.set_xlabel("avg firing rate [Hz]")

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_gridsearch(mods, param, other_mods=dict(), logscale=True, path=""):
    "Plot hyperparameter search results and compare to other (best) models"

    df = pd.DataFrame(mods.cv_results_)
    ncv = len([c for c in df.columns if c.startswith("split")])
    x = [np.array(*i.values()).item() for i in df.loc[:, "params"]]
    y, yerr = df.loc[:, "mean_test_score"], df.loc[:, "std_test_score"] / np.sqrt(ncv)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr, label=param)
    ax.axvline(x[y.argmax()], c="gray", ls=":")

    for i, (name, mods) in enumerate(other_mods.items()):
        df = pd.DataFrame(mods.cv_results_)
        ncv = len([c for c in df.columns if c.startswith("split")])
        i_max = df.loc[:, "mean_test_score"].argmax()
        ds = df.loc[i_max, :]
        x = np.array(*ds.loc["params"].values()).item()
        y, yerr = ds.loc["mean_test_score"], ds.loc["std_test_score"] / np.sqrt(ncv)
        ax.axhline(y, lw=1, label=name, c=f"C{i+1}")
        ax.axhline(y + yerr, ls=":", lw=1, c=f"C{i+1}")
        ax.axhline(y - yerr, ls=":", lw=1, c=f"C{i+1}")

    ax.legend()
    ax.set_title(f"best score: {y.max():1.2f}")
    ax.set_xlabel("parameter")
    ax.set_ylabel("score")

    if logscale:
        ax.set_xscale("log")

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_missing(df_psth, bin_size, vmax=None, path=""):
    df = pd.pivot_table(data=df_psth, values="fr", index="unit", columns="T")
    # df = df.apply(lambda x: x / np.mean(x), axis=1)

    n_unt, n_bins = df.shape
    fig, axarr = plt.subplots(
        ncols=2, width_ratios=(3, 1), figsize=(n_bins / 0.5e4, n_unt / 10)
    )

    ax = axarr[0]
    im = ax.pcolormesh(df.columns * bin_size, df.index, df.values, vmax=vmax)

    label = "firing rate [Hz]"
    if vmax:
        label = label + f" (capped at {vmax})"

    fig.colorbar(im, ax=ax, location="right", orientation="vertical", label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("unit")
    # for b in B:
    #     ax.axvline(b * bin_size, lw=.1, c='gray')

    ax = axarr[1]
    r_nan = (df != df).sum(axis=1) / df.shape[1]
    y, x = r_nan.index, r_nan.values * 100
    ax.barh(y, x)
    ax.margins(y=0)
    ax.set_xlim((0, 100))
    ax.set_ylabel("unit")
    ax.set_xlabel("missing [%]")

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_unit(rec, bin_size, unit, xlims=(None, None), path=""):
    fig, ax = plt.subplots(figsize=(20, 5))

    d = rec.df_prec.groupby("unit").get_group(unit)
    x = d.loc[:, "T"].values * bin_size
    y = d.loc[:, "fr"].values

    rec.df_trl.loc[:, "Tf"] = np.cumsum(
        rec.df_trl.loc[:, "dtf"] - rec.df_trl.loc[:, "dt0"]
    )
    for trl in d.loc[:, "trial"].unique():
        tf = rec.df_trl.groupby("trial").get_group(trl).loc[:, "Tf"].item()
        ax.axvline(tf, c="gray", lw=0.5)

    ax.plot(x, y)
    ax.margins(x=0)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("firing rate [Hz]")
    ax.set_xlim(xlims)

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_mean_response(df_bin, arr_pred=None, scores={}, path=""):
    "Plot PSTH for all units in `df_bin` and optionally add predictions in `arr_pred`"

    if arr_pred is not None:
        df_bin_pred = df_bin.copy()
        df_bin_pred.loc[:, :] = arr_pred

    nu = len(df_bin.columns)
    nc = 5
    nr = int(np.ceil(nu / nc))

    fig, axmat = plt.subplots(
        ncols=nc, nrows=nr, figsize=(3 * nc, 2 * nr), squeeze=False
    )

    for ax, u in zip(axmat.flatten(), df_bin.columns):
        df = df_bin.loc[:, u].reset_index()
        df.loc[:, "type"] = "true"
        if arr_pred is not None:
            df_pred = df_bin_pred.loc[:, u].reset_index()
            df_pred.loc[:, "type"] = "pred"
            df = pd.concat([df, df_pred])

        sns.lineplot(
            ax=ax, data=df, x="bin", y=u, hue="type", errorbar="sd", legend=False
        )

        title = f"{u[0]} {u[1]}"
        if scores:
            title += f" ({scores[u]:1.3f})"
        ax.set_title(title)
        ax.margins(x=0)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axmat[-1]:
        ax.set_xlabel("time from cue [s]")
    for ax in axmat.T[0]:
        ax.set_ylabel("firing rate [Hz]")

    for ax in axmat.flatten():
        if not ax.has_data():
            ax.set_axis_off()

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def _barplot_wrapper(df, col, ax):
    # plot trial type counts
    sns.histplot(data=df, y=col, ax=ax, discrete=True, stat="percent", multiple="stack", shrink=0.8)

    # absolute and relative counts as labels
    c = ax.containers[0]
    c_rel = c.datavalues
    c_abs = c_rel * len(df) / 100
    l = [f" {int(i):d} ({j:1.1f} %)" for i, j in zip(c_abs, c_rel)]
    ax.bar_label(c, labels=l)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of trials")

    return ax


def plot_trial_infos(df_trl, path=""):
    "Plot trial type, response, and lick time distributions"

    df = df_trl.copy()
    df.loc[:, "short"] = df.loc[:, "trial_type"].str.split("_").str[:2].str.join("_")

    # plot
    fig, axmat = plt.subplots(figsize=(14, 6), nrows=2, ncols=2)
    axarr = axmat.T.flatten()

    ax = axarr[0]
    ax.set_title("trial type distribution")
    _barplot_wrapper(df, "short", ax)
    ax.set_ylabel("Trial type")

    if "dt_lck_group" in df.columns:
    
        ax = axarr[1]
        ax.set_title("trial response distribution")
        _barplot_wrapper(df, "response", ax)
        ax.set_ylabel("Trial response")

        ax = axarr[2]
        ax.set_title("lick time distribution")
        sns.histplot(
            data=df, x="dt_lck", hue="response", ax=ax, multiple="layer", binwidth=0.2
        )
        ax.set_xlabel("lick time relative to cue (s)")

        ax = axarr[3]
        ax.set_title("lick groups")
        sns.histplot(data=df, x="dt_lck_group", ax=ax, hue="response", multiple="stack", shrink=0.8)
    
    else:
        ax = axarr[1]
        ax.set_title("water raio distribution")
        _barplot_wrapper(df, "water_ratio", ax)
        ax.set_ylabel("Water ratio")

        ax = axarr[2]
        x = df.loc[:, 'trial']
        y = df.loc[:, 'water_ratio']
        ax.plot(x, y, label='water ratio')
        
        y = df.loc[:, 'reward_delay']
        ax.plot(x, y, label='reward delay')

        label = 'no reward'
        for trial in df.loc[y != y, 'trial']:
            ax.axvspan(trial, trial+1, color='gray', alpha=0.2, label=label)
            label = None
        ax.set_xlabel('Trial')
        ax.set_ylabel('Water ratio / Reward delay [s]')
        ax.legend()
        ax.grid(False)
        ax.margins(x=0)

        ax = axarr[3]
        ds = df.loc[:, 'trial_block'].astype('category')
        cat_order = [ 'base line', 'base ratio', 'large ratio test', 'no delay', 'delay test' ]
        for cat in ds.cat.categories:
            if cat not in cat_order:
                ds = ds.cat.remove_categories(cat)
        ds.cat.reorder_categories(cat_order, ordered=True)
        df.loc[:, 'trial_block'] = ds
        sns.histplot(data=df, ax=ax, x="trial_block", hue="response", multiple="stack", shrink=0.8)

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_box_and_points(df, x, y, col=None, hue=None, order=[], hue_order=[], ylim=(-1, 1), path=""):
    "Plot boxplots with points on top"

    if not hue_order:
        hue_order = sorted(df.loc[:, hue].unique())
    if not order:
        order = sorted(df.loc[:, x].unique())

    g = sns.catplot(
        df,
        x=x,
        y=y,
        col=col,
        hue=hue,
        order=order,
        hue_order=hue_order,
        kind="box",
        whis=0,
        fliersize=0,
        palette="pastel",
        sharex=False,
        facet_kws={"ylim": ylim},
        legend=True
    )

    g.map(
        sns.stripplot,
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        dodge=True,
        palette="deep",
        edgecolor="auto",
        linewidth=0.5,
        size=1,
    )

    sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
    
    fig = g.fig
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_opt_ranks(df, path=""):
    "Plot best scores vs optimal ranks"

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(data=df, x="rank_opt", y="score_best", hue="interaction", ax=ax)
    ax.set_xlabel("rank_opt")
    ax.set_ylabel("score_best")

    # Anchor legend at the top right outside the plot
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)

def plot_mode(df_mode, group=False, path=""):

    fig, ax = plt.subplots()
    if group:
        hue = df_mode.columns[df_mode.columns.str.endswith('_group')].item()
    else:
        hue = None
    sns.lineplot(ax=ax, data=df_mode, x='bin', y='mode_activity', hue=hue)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mode activity')

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


