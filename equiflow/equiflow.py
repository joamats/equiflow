
"""
The equiflow package is used for creating "Equity-focused Cohort Section Flow Diagrams"
for cohort selection in clinical and machine learning papers.
"""

import numpy as np
import pandas as pd

class TableZero:
  def __init__(self,
               dfs: list,
               cols: list,
               decimals: int=3,
               normalize: bool=True):

    if not isinstance(dfs, list) or len(dfs) < 1:
        raise ValueError("dfs must be a list with length ≥ 1")

    if not isinstance(cols, list):
        raise ValueError("cols must be a list")

    self.data = dfs
    self.columns = cols
    self.decimals = decimals
    self.normalize = normalize

    self.table = pd.DataFrame()


  def get_original_uniques(self):

    self.original_uniques = dict()

    # get uniques values ignoring NaNs
    for c in self.columns:
      self.original_uniques[c] = self.data[0][c].dropna().unique()



  def my_value_counts(self, df, col) -> pd.DataFrame():

    o_uniques = self.original_uniques[col]
    counts = pd.DataFrame(columns=[col], index=o_uniques)
    # o_uniques = np.insert(o_uniques, 0, None)
    n = len(df)

    for k, o in enumerate(o_uniques):
      if self.normalize:
        counts.loc[o,col] = ((df[col] == o).sum() / n).round(self.decimals)
      else:
        counts.loc[o,col] = (df[col] == o).sum()

    return counts #.rename(index={None: '[ Missing ]'})


  def table_one(self):

    self.get_original_uniques()

    for i, df in enumerate(self.data):

      df_counts = pd.DataFrame()

      # add n counts
      for col in self.columns:

        counts = self.my_value_counts(df, col)

        # display(counts)

        melted_counts = pd.melt(counts.reset_index(), id_vars=['index']) \
                          .set_index(['variable','index'])



        df_counts = pd.concat([df_counts, melted_counts], axis=0)
        df_counts.loc[(col,'Missingness'),'value'] = df[col].isnull().sum()

      df_counts.rename(columns={'value': i}, inplace=True)
      self.table = pd.concat([self.table, df_counts], axis=1)

    # add super header
    self.table = self.table.set_axis(
        pd.MultiIndex.from_product([['Cohort'], self.table.columns]),
        axis=1)


    # renames
    self.table.index.names = ['Variable', 'Value']

    return self.table

