import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def normalization(arr: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(arr)

def dims_reduction(df: pd.DataFrame, features: list[str], dims: int) -> pd.DataFrame:
    x = df.loc[:, features].values
    y = df.loc[:,['Атака']].values

    pca = PCA(n_components=dims)

    components = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = components, 
                               columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['Атака']]], axis = 1)
    return finalDf

def split_by_blocks(arr: np.ndarray, num_blocks: int) -> list:
    b1 = np.split(
        arr[0],
        num_blocks
    )
    b2 = np.split(
        arr[1],
        num_blocks
    )
    return [block for block in zip(b1, b2)]
    