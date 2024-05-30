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
               decimals: int=1,
               format: str='N',
               missingness: bool=True,
               ):

    if not isinstance(dfs, list) or len(dfs) < 1:
        raise ValueError("dfs must be a list with length ≥ 1")

    if not isinstance(cols, list):
        raise ValueError("cols must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    if format not in ['%', 'N', 'N (%)']:
        raise ValueError("format must be '%', 'N', or 'N (%)'")
    
    if not isinstance(missingness, bool):
        raise ValueError("missingness must be a boolean")

    self.data = dfs
    self.columns = cols
    self.decimals = decimals
    self.format = format
    self.missingness = missingness


  def __get_original_uniques(self):

    self.original_uniques = dict()

    # get uniques values ignoring NaNs
    for c in self.columns:
      self.original_uniques[c] = self.data[0][c].dropna().unique()



  def __my_value_counts(self, df, col) -> pd.DataFrame(): # type: ignore

    o_uniques = self.original_uniques[col]
    counts = pd.DataFrame(columns=[col], index=o_uniques)

    # get the number of observations, based on whether we want to include missingness
    if self.missingness:
      n = len(df)
    else:
      n = len(df) - df[col].isnull().sum() # denominator will be the number of non-missing observations

    for o in o_uniques:
      if self.format == '%':
        counts.loc[o,col] = ((df[col] == o).sum() / n * 100).round(self.decimals)
  
      elif self.format == 'N':
        counts.loc[o,col] = (df[col] == o).sum()
   
      elif self.format == 'N (%)':
        n_counts = (df[col] == o).sum()
        perc_counts = (n_counts / n * 100).round(self.decimals)
        counts.loc[o,col] = f"{n_counts} ({perc_counts})"

      else:
        raise ValueError("format must be '%', 'N', or 'N (%)'")


    return counts 
  

  def __add_missing_counts(self, df, col, df_counts) -> pd.DataFrame(): # type: ignore

    n = len(df)

    if self.format == '%':
      df_counts.loc[(col,'Missing'),'value'] = (df[col].isnull().sum() / n * 100).round(self.decimals)
    
    elif self.format == 'N':
      df_counts.loc[(col,'Missing'),'value'] = df[col].isnull().sum()

    elif self.format == 'N (%)':
      n_missing = df[col].isnull().sum()
      perc_missing = df[col].isnull().sum() / n * 100
      df_counts.loc[(col,'Missing'),'value'] = f"{n_missing} ({(perc_missing).round(self.decimals)})"

    else:
      raise ValueError("format must be '%', 'N', or 'N (%)'")

    return df_counts
  
  
  def __add_overall_counts(self, df, df_counts) -> pd.DataFrame(): # type: ignore

    df_counts.loc[('Overall', ' '), 'value'] = f"{len(df)}"

    return df_counts


  # change name to represent the fact that this view is for the distribution of the cohorts
  def view_cohorts(self):

    table = pd.DataFrame()

    self.__get_original_uniques()

    for i, df in enumerate(self.data):

      df_counts = pd.DataFrame()

      for col in self.columns:

        counts = self.__my_value_counts(df, col)

        melted_counts = pd.melt(counts.reset_index(), id_vars=['index']) \
                          .set_index(['variable','index'])

        df_counts = pd.concat([df_counts, melted_counts], axis=0)

        if self.missingness:
          df_counts = self.__add_missing_counts(df, col, df_counts)
        
      df_counts = self.__add_overall_counts(df, df_counts)
      

      df_counts.rename(columns={'value': i}, inplace=True)
      table = pd.concat([table, df_counts], axis=1)

    # add super header
    table = table.set_axis(
        pd.MultiIndex.from_product([['Cohort'], table.columns]),
        axis=1)

    # renames indexes
    table.index.names = ['Variable', 'Value']

    # reorder values of "Variable" (level 0) such that 'Overall' comes first
    table = table.sort_index(level=0, key=lambda x: x == 'Overall',
                                       ascending=False, sort_remaining=False)

    return table
  
  # to-do: add a view for the flow of the cohorts, with N, remove, new N
  def view_flow(self):

    table = pd.DataFrame(columns=['Flow', '', 'N',])
    rows = []

    for i in range(len(self.data) - 1):

      df_0 = self.data[i]
      df_1 = self.data[i+1]
      label = f"{i} to {i+1}"

      rows.append({'Flow': label,
                   '': 'Inital, n',
                   'N': len(df_0)})
      
      rows.append({'Flow': label,
                   '': 'Removed, n',
                   'N': len(df_0) - len(df_1)})
      
      rows.append({'Flow': label,
                   '': 'Result, n',
                   'N': len(df_1)})

    table = pd.DataFrame(rows)

    # pivot table to have the flow as the columns
    table = table.pivot(index='', columns='Flow', values='N')

    return table
      
      




    pass

