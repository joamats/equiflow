"""
The equiflow package is used for creating "Equity-focused Cohort Section Flow Diagrams"
for cohort selection in clinical and machine learning papers.
"""

import numpy as np
import pandas as pd

class TableZero:
  def __init__(self,
               dfs: list,
               decimals: int=1,
               ):

    if not isinstance(dfs, list) or len(dfs) < 1:
        raise ValueError("dfs must be a list with length ≥ 1")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")

    self.data = dfs
    self.decimals = decimals

    self.__clean_missing()


  # method to categorize missing values under the same label
  def __clean_missing(self): 
    
    for i, df in enumerate(self.data):

      # map all missing values possibilities to Null
      df = df.replace(['', ' ', 'NA', 'N/A', 'na', 'n/a',
                      'NA ', 'N/A ', 'na ', 'n/a ', 'NaN',
                      'nan', 'NAN', 'Nan', 'N/A;', '<NA>',
                      "_", '__', '___', '____', '_____',
                      'NaT', 'None', 'none', 'NONE', 'Null',
                      'null', 'NULL', 'missing', 'Missing',
                      np.nan, pd.NA], None)
      
      # replace
      self.data[i] = df


  # method to get the unique values, before any exclusion (at i=0)
  def __get_original_uniques(self, cols):

    original_uniques = dict()

    # get uniques values ignoring NaNs
    for c in cols:
      original_uniques[c] = self.data[0][c].dropna().unique()

    return original_uniques


  # method to get the value counts for a given column
  def __my_value_counts(self,
                        df: pd.DataFrame(),
                        original_uniques: dict,
                        col: str,
                        missingness: bool=True,
                        format: str='N',
                        ) -> pd.DataFrame(): # type: ignore

    o_uniques = original_uniques[col]
    counts = pd.DataFrame(columns=[col], index=o_uniques)

    # get the number of observations, based on whether we want to include missingness
    if missingness:
      n = len(df)
    else:
      n = len(df) - df[col].isnull().sum() # denominator will be the number of non-missing observations

    for o in o_uniques:
      if format == '%':
        counts.loc[o,col] = ((df[col] == o).sum() / n * 100).round(self.decimals)
  
      elif format == 'N':
        counts.loc[o,col] = (df[col] == o).sum()
   
      elif format == 'N (%)':
        n_counts = (df[col] == o).sum()
        perc_counts = (n_counts / n * 100).round(self.decimals)
        counts.loc[o,col] = f"{n_counts} ({perc_counts})"

      else:
        raise ValueError("format must be '%', 'N', or 'N (%)'")


    return counts 
  

  # method to add missing counts to the table
  def __add_missing_counts(self,
                           df: pd.DataFrame(),
                           col: str,
                           format: str,
                           df_counts: pd.DataFrame(),
                           ) -> pd.DataFrame(): # type: ignore

    n = len(df)

    if format == '%':
      df_counts.loc[(col,'Missing'),'value'] = (df[col].isnull().sum() / n * 100).round(self.decimals)
    
    elif format == 'N':
      df_counts.loc[(col,'Missing'),'value'] = df[col].isnull().sum()

    elif format == 'N (%)':
      n_missing = df[col].isnull().sum()
      perc_missing = df[col].isnull().sum() / n * 100
      df_counts.loc[(col,'Missing'),'value'] = f"{n_missing} ({(perc_missing).round(self.decimals)})"

    else:
      raise ValueError("format must be '%', 'N', or 'N (%)'")

    return df_counts
  
  
  # method to add overall counts to the table
  def __add_overall_counts(self,
                           df,
                           df_counts
                           ) -> pd.DataFrame(): # type: ignore

    df_counts.loc[('Overall', ' '), 'value'] = f"{len(df)}"

    return df_counts


  # first view: cohort flow numbers
  def view_flow(self):
    
    table = pd.DataFrame(columns=['Cohort Flow', '', 'N',])
    rows = []

    for i in range(len(self.data) - 1):

      df_0 = self.data[i]
      df_1 = self.data[i+1]
      label = f"{i} to {i+1}"

      rows.append({'Cohort Flow': label,
                   '': 'Inital, n',
                   'N': len(df_0)})
      
      rows.append({'Cohort Flow': label,
                   '': 'Removed, n',
                   'N': len(df_0) - len(df_1)})
      
      rows.append({'Cohort Flow': label,
                   '': 'Result, n',
                   'N': len(df_1)})

    table = pd.DataFrame(rows)

    table = table.pivot(index='', columns='Cohort Flow', values='N')

    return table


  # second view: cohort flow distributions
  def view_cohorts(self,
                   cols: list,
                   decimals: int=1,
                   format: str='N',
                   missingness: bool=True, 
      ):

    if not isinstance(cols, list):
        raise ValueError("cols must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    if format not in ['%', 'N', 'N (%)']:
        raise ValueError("format must be '%', 'N', or 'N (%)'")
    
    if not isinstance(missingness, bool):
        raise ValueError("missingness must be a boolean")
    
    self.decimals = decimals

    table = pd.DataFrame()

    original_uniques = self.__get_original_uniques(cols)

    for i, df in enumerate(self.data):

      df_counts = pd.DataFrame()

      for col in cols:

        counts = self.__my_value_counts(df, original_uniques, col, missingness, format)

        melted_counts = pd.melt(counts.reset_index(), id_vars=['index']) \
                          .set_index(['variable','index'])

        df_counts = pd.concat([df_counts, melted_counts], axis=0)

        if missingness:
          df_counts = self.__add_missing_counts(df, col, format, df_counts)
        
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
  
  
  # third view: cohort flow distribution differences
  def view_differences(self):
    pass
      
      