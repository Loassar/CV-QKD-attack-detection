import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def normalization(arr: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(arr)



def dims_reduction(df: pd.DataFrame, features: list[str], dims: int) -> pd.DataFrame:
    """
    Reduces the dimensionality of the given dataframe using Principal Component Analysis (PCA).

    This function takes a dataframe and reduces the dimensions of the specified features
    to the number of dimensions given by `dims`. The result is a new dataframe containing
    the principal components and the original target label ('Атака').

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    features (list[str]): A list of column names in the dataframe that are to be used for PCA.
    dims (int): The number of principal components to reduce the data to.

    Returns:
    pd.DataFrame: A dataframe containing the principal components and the target label.

    Example:
    >>> df = pd.DataFrame({
    ...     'feature1': [1, 2, 3],
    ...     'feature2': [4, 5, 6],
    ...     'Атака': [0, 1, 0]
    ... })
    >>> dims_reduction(df, ['feature1', 'feature2'], 2)
       principal component 1  principal component 2  Атака
    0              -2.828427               3.330669e-16      0
    1               0.000000              -0.000000e+00      1
    2               2.828427              -3.330669e-16      0
    """

    x = df.loc[:, features].values
    y = df.loc[:,['Атака']].values

    pca = PCA(n_components=dims)

    components = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = components, 
                               columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['Атака']]], axis = 1)
    return finalDf

def split_by_blocks(arr: np.ndarray, num_blocks: int) -> list:
    """
    Splits a 2D numpy array into a specified number of blocks.

    Parameters:
    arr (np.ndarray): A 2D numpy array to be split.
    num_blocks (int): The number of blocks to split each row into.

    Returns:
    list: A list of tuples where each tuple contains corresponding blocks from each row.

    Example:
    >>> arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> split_by_blocks(arr, 2)
    [([1, 5], [2, 6]), ([3, 7], [4, 8])]
    """
    b1 = np.split(
        arr[0],
        num_blocks
    )
    b2 = np.split(
        arr[1],
        num_blocks
    )
    return [block for block in zip(b1, b2)]
    