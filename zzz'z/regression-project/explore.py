from imports import *

def plot_variable_pairs(df):
	sns.pairplot(df.sample(100), kind='reg', diag_kind='hist', palette='icefire', plot_kws={'line_kws':{'color':'red'}})
	return plt.show()


def plot_categorical_and_continuous_vars(
    df: pd.core.frame.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str]
) -> None:
    '''
        Plot a boxplot, barplot, and histplot for each combination of continous 
        and categorical column in the dataframe provided.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the data to be plotted.
        categorical_cols: list[str]
            A list of the categorical columns to plot.
        continuous_cols: list[str]
            A list of the continuous columns to plot.
    '''

    for con in continuous_cols:
        for cat in categorical_cols:
            fig = plt.figure(figsize = (14, 4))
            fig.suptitle(f'{con} v. {cat}')

            plt.subplot(1, 3, 1)
            sns.boxplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 2)
            sns.barplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 3)
            sns.histplot(data = df, x = con, bins = 10, hue = cat)
            plt.show()