import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def null_checker(df, sort=True, ascending=False):
    """
    Mengecek missing value dengan output dataframe dengan kolom jumlah missing value dan presentasenya.

    Parameters
    ----------
    df : dataframe
        Dataframe yang ingin dilakukan pengecekan missing value.

    sort : boolean (default = True)
        Sortir berdasarkarkan jumlah missing value.

    ascending : boolean (default = False)

    Returns
    -------
    Dataframe yang menunjukkan jumlah missing value dengan presentase (pembulatan 2 desimal)

    """
    null_cols = {
        "null (sum)": df.isna().sum().values,
        "null (%)": df.isna().sum().values / df.shape[0] * 100,
    }
    null_cols = pd.DataFrame(data=null_cols, index=df.isna().sum().index)

    if sort == True:
        if ascending == False:
            null_cols = round(
                null_cols.sort_values(by="null (sum)", ascending=False), 2
            )
        else:
            null_cols = round(null_cols.sort_values(by="null (sum)", ascending=True), 2)

        return null_cols

    else:
        return round(null_cols, 2)


def countplot_annot(nrow, ncol, columns, data, rotate=None, rcol=None, t_height=25):
    """
    Function untuk ploting sns.counplot dengan penambahan presentase
    di atas bar. (Versi tanpa hue)
    """
    for index, col in enumerate(columns):
        plt.subplot(nrow, ncol, index + 1)

        order = sorted(data[col].unique())
        ax = sns.countplot(data=data, x=col, order=order)
        ax.set_ylabel("")

        if rotate != None:
            if col in rcol:
                plt.xticks(rotation=rotate)

        total = len(data)
        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width() / 2.0,
                p.get_height() + t_height,
                "{:.1f}%".format(100 * p.get_height() / total),
                ha="center",
            )


def countplot_annot_hue(
    nrow, ncol, columns, hue, data, rotate=None, rcol=None, t_height=30
):
    """
    Function untuk ploting sns.counplot dengan penambahan presentase
    di atas bar. (Versi dengan hue)
    """
    assert hue.nunique() == 2, "Hanya bisa plotting menggunakan hue dengan 2 class"

    for index, col in enumerate(columns):
        plt.subplot(nrow, ncol, index + 1)

        order = sorted(data[col].unique())
        ax = sns.countplot(data=data, x=col, hue=hue, order=order)
        ax.set_ylabel("")

        if rotate != None:
            if col in rcol:
                plt.xticks(rotation=rotate)

        bars = ax.patches
        half = int(len(bars) / 2)
        left_bars = bars[:half]
        right_bars = bars[half:]

        for left, right in zip(left_bars, right_bars):
            height_l = left.get_height()
            height_r = right.get_height()
            total = height_l + height_r

            ax.text(
                left.get_x() + left.get_width() / 2.0,
                height_l + t_height,
                "{0:.0%}".format(height_l / total),
                ha="center",
            )
            ax.text(
                right.get_x() + right.get_width() / 2.0,
                height_r + t_height,
                "{0:.0%}".format(height_r / total),
                ha="center",
            )
