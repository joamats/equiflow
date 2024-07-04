"""
The equiflow package is used for creating "Equity-focused Cohort Section Flow Diagrams"
for cohort selection in clinical and machine learning papers.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


class EquiFlow:
  def __init__(self,
               dfs: list) -> None:
    
    if not isinstance(dfs, list) or len(dfs) < 1:
      raise ValueError("dfs must be a list with length ≥ 1")
    
    self._dfs = dfs

    self.__clean_missing()

    # to-do: add alternative construction if dfs is a single dataframe


  # method to categorize missing values under the same label
  def __clean_missing(self): 
    
    for i, df in enumerate(self._dfs):

      # map all missing values possibilities to Null
      df = df.replace(['', ' ', 'NA', 'N/A', 'na', 'n/a',
                      'NA ', 'N/A ', 'na ', 'n/a ', 'NaN',
                      'nan', 'NAN', 'Nan', 'N/A;', '<NA>',
                      "_", '__', '___', '____', '_____',
                      'NaT', 'None', 'none', 'NONE', 'Null',
                      'null', 'NULL', 'missing', 'Missing',
                      np.nan, pd.NA], None)
  
    self._dfs[i] = df
      

 
  def add_exclusion(self, mask, label):
    pass

  def table_flows(self, *args, **kwargs):
    table = TableFlows(self._dfs, *args, **kwargs)
    self._table_flows = table.build()
    return self._table_flows

  def table_characteristics(self, *args, **kwargs) -> pd.DataFrame:
    table = TableCharacteristics(self._dfs, *args, **kwargs)
    self._table_characteristics = table.build()
    return self._table_characteristics

  def table_drifts(self, *args, **kwargs) -> pd.DataFrame:
    # based on the arguments that we have, build auxiliary tables
    table_cat_n_kwargs = kwargs.copy()
    table_cat_n_kwargs['normal'] = []
    table_cat_n_kwargs['nonnormal'] = []
    table_cat_n_kwargs['format_cat'] = 'N'
    table_cat_n = TableCharacteristics(self._dfs, *args, **table_cat_n_kwargs).build()

    table_cat_perc_kwargs = kwargs.copy()
    table_cat_perc_kwargs['normal'] = []
    table_cat_perc_kwargs['nonnormal'] = []
    table_cat_perc_kwargs['format_cat'] = '%'
    table_cat_perc = TableCharacteristics(self._dfs, *args, **table_cat_perc_kwargs).build()

    table_cont_mean_kwargs = kwargs.copy()
    # table_cat_perc_kwargs['categorical'] = [] ## currently generates a bug, need to fix!!
    table_cont_mean_kwargs['normal'] = table_cont_mean_kwargs['normal'] + table_cont_mean_kwargs['nonnormal']
    table_cont_mean_kwargs['nonnormal'] = []
    table_cont_mean_kwargs['format_cont'] = 'Mean'
    table_cont_mean = TableCharacteristics(self._dfs, *args, **table_cont_mean_kwargs).build()

    table_cont_sd_kwargs = kwargs.copy()
    # table_cat_perc_kwargs['categorical'] = [] ## currently generates a bug, need to fix!!
    table_cont_sd_kwargs['normal'] = table_cont_sd_kwargs['normal'] + table_cont_sd_kwargs['nonnormal']
    table_cont_sd_kwargs['nonnormal'] = []  
    table_cont_sd_kwargs['format_cont'] = 'SD'
    table_cont_sd = TableCharacteristics(self._dfs, *args, **table_cont_sd_kwargs).build()

    table = TableDrifts(self._dfs,
                        self._table_flows,
                        self._table_characteristics,
                        table_cat_n,
                        table_cat_perc,
                        table_cont_mean,
                        table_cont_sd,
                        )
    return table#.build()
  

  def plot_flows(self):
    pass

  def write_report(self):
    pass


class TableFlows:
  def __init__(
        self,
        dfs: list,
        label_suffix: Optional[bool] = True,
        thousands_sep: Optional[bool] = True,
    ) -> pd.DataFrame:

    if not isinstance(dfs, list) or len(dfs) < 1:
      raise ValueError("dfs must be a list with length ≥ 1")
    
    if not isinstance(label_suffix, bool):
      raise ValueError("label_suffix must be a boolean")
    
    self._dfs = dfs
    self._label_suffix = label_suffix
    self._thousands_sep = thousands_sep

    self.table = self._build()

    return self.table

  def _build(self):

    table = pd.DataFrame(columns=['Cohort Flow', '', 'N',])
    rows = []

    for i in range(len(self._dfs) - 1):

      df_0 = self._dfs[i]
      df_1 = self._dfs[i+1]
      label = f"{i} to {i+1}"

      if self._label_suffix:
        suffix = ', n'

      else:
        suffix = ''

      if self._thousands_sep:
        n0_string = f"{len(df_0):,}"
        n1_string = f"{len(df_0) - len(df_1):,}"
        n2_string = f"{len(df_1):,}"

      else:
        n0_string = len(df_0)
        n1_string = len(df_0) - len(df_1)
        n2_string = len(df_1)


      rows.append({'Cohort Flow': label,
                   '': 'Inital' + suffix,
                   'N': n0_string})
      
      rows.append({'Cohort Flow': label,
                   '': 'Removed' + suffix,
                   'N': n1_string})
      
      rows.append({'Cohort Flow': label,
                   '': 'Result' + suffix,
                   'N': n2_string})

    table = pd.DataFrame(rows)

    table = table.pivot(index='', columns='Cohort Flow', values='N')

    return table
  

class TableCharacteristics:
  def __init__(
      self,
      dfs: list,
      categorical: Optional[list] = None,
      normal: Optional[list] = None,
      nonnormal: Optional[list] = None,
      decimals: Optional[int] = 1,
      format_cat: Optional[str] = 'N (%)',
      format_normal: Optional[str] = 'Mean ± SD',
      format_nonnormal: Optional[str] = 'Median [IQR]',
      thousands_sep: Optional[bool] = True,
      missingness: Optional[bool] = True,
      label_suffix: Optional[bool] = True,
      rename: Optional[dict] = None,
  ) -> pd.DataFrame:
        
    if not isinstance(dfs, list) or len(dfs) < 1:
        raise ValueError("dfs must be a list with length ≥ 1")
    
    if not isinstance(categorical, list):
        raise ValueError("categorical must be a list")

    if not isinstance(normal, list):
        raise ValueError("normal must be a list")
    
    if not isinstance(nonnormal, list):
        raise ValueError("nonnormal must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    if format_cat not in ['%', 'N', 'N (%)']:
        raise ValueError("format must be '%', 'N', or 'N (%)'")
    
    if format_normal not in ['Mean ± SD', 'Mean', 'SD']:
        raise ValueError("format must be 'Mean ± SD' or 'Mean' or 'SD'")
    
    if format_nonnormal not in ['Median [IQR]', 'Mean', 'SD']:
        raise ValueError("format must be 'Median [IQR]' or 'Mean' or 'SD'")
    
    if not isinstance(thousands_sep, bool):
        raise ValueError("thousands_sep must be a boolean")
    
    if not isinstance(missingness, bool):
        raise ValueError("missingness must be a boolean")
    
    if not isinstance(label_suffix, bool):
        raise ValueError("label_suffix must be a boolean")
    
    if not isinstance(rename, dict):
        raise ValueError("rename must be a dictionary")
    
    self._dfs = dfs
    self._categorical = categorical
    self._normal = normal
    self._nonnormal = nonnormal
    self._decimals = decimals
    self._missingness = missingness
    self._format_cat = format_cat
    self._format_normal = format_normal
    self._format_nonnormal = format_nonnormal
    self._thousands_sep = thousands_sep
    self._label_suffix = label_suffix
    self._rename = rename

    self.table = self._build()

    return self.table

  # method to get the unique values, before any exclusion (at i=0)
  def _get_original_uniques(self, cols):

    original_uniques = dict()

    # get uniques values ignoring NaNs
    for c in cols:
      original_uniques[c] = self._dfs[0][c].dropna().unique()

    return original_uniques


  # method to get the value counts for a given column
  def _my_value_counts(self,
                       df: pd.DataFrame(),
                       original_uniques: dict,
                       col: str,
                      ) -> pd.DataFrame(): # type: ignore

    o_uniques = original_uniques[col]
    counts = pd.DataFrame(columns=[col], index=o_uniques)

    # get the number of observations, based on whether we want to include missingness
    if self._missingness:
      n = len(df)
    else:
      n = len(df) - df[col].isnull().sum() # denominator will be the number of non-missing observations

    for o in o_uniques:
      if self._format_cat == '%':
        counts.loc[o,col] = ((df[col] == o).sum() / n * 100).round(self._decimals)
  
      elif self._format_cat == 'N':
        if self._thousands_sep:
          counts.loc[o,col] = f"{(df[col] == o).sum():,}"
        else:
          counts.loc[o,col] = (df[col] == o).sum()
   
      elif self._format_cat == 'N (%)':
        n_counts = (df[col] == o).sum()
        perc_counts = (n_counts / n * 100).round(self._decimals)
        if self._thousands_sep:
          counts.loc[o,col] = f"{n_counts:,} ({perc_counts})"
        else:
          counts.loc[o,col] = f"{n_counts} ({perc_counts})"

      else:
        raise ValueError("format must be '%', 'N', or 'N (%)'")

    return counts 
  
  # method to report distribution of normal variables
  def _normal_vars_dist(self,
                        df: pd.DataFrame(),
                        col: str,
                        df_dists: pd.DataFrame(),
                        ) -> pd.DataFrame():
    
    df.loc[:,col] = pd.to_numeric(df[col], errors='raise')
    
    if self._format_normal == 'Mean ± SD':
      col_mean = np.round(df[col].mean(), self._decimals)
      col_std = np.round(df[col].std(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = f"{col_mean} ± {col_std}"

    elif self._format_normal == 'Mean':
      col_mean = np.round(df[col].mean(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_mean
    
    elif self._format_normal == 'SD':
      col_std = np.round(df[col].std(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_std

    return df_dists
  
  def _nonnormal_vars_dist(self,
                           df: pd.DataFrame(),
                           col: str,
                           df_dists: pd.DataFrame(),
                          ) -> pd.DataFrame():
     
    df.loc[:,col] = pd.to_numeric(df[col], errors='raise')

    if self._format_nonnormal == 'Mean':
      col_mean = np.round(df[col].mean(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_mean

    elif self._format_nonnormal == 'Median [IQR]':
      col_median = np.round(df[col].median(), self._decimals)
      col_q1 = np.round(df[col].quantile(0.25), self._decimals)
      col_q3 = np.round(df[col].quantile(0.75), self._decimals)

      df_dists.loc[(col, ' '), 'value'] = f"{col_median} [{col_q1}, {col_q3}]"

    elif self._format_nonnormal == 'SD':
      col_std = np.round(df[col].std(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_std

    return df_dists
  

  # method to add missing counts to the table
  def _add_missing_counts(self,
                           df: pd.DataFrame(),
                           col: str,
                           df_dists: pd.DataFrame(),
                           ) -> pd.DataFrame(): # type: ignore

    n = len(df)

    if self._format_cat == '%':
      df_dists.loc[(col,'Missing'),'value'] = (df[col].isnull().sum() / n * 100).round(self._decimals)
    
    elif self._format_cat == 'N':
      if self._thousands_sep:
        df_dists.loc[(col,'Missing'),'value'] = f"{df[col].isnull().sum():,}"
      else:
        df_dists.loc[(col,'Missing'),'value'] = df[col].isnull().sum()

    elif self._format_cat == 'N (%)':
      n_missing = df[col].isnull().sum()
      perc_missing = df[col].isnull().sum() / n * 100
      if self._thousands_sep:
        df_dists.loc[(col,'Missing'),'value'] = f"{n_missing:,} ({(perc_missing).round(self._decimals)})"
      else: 
        df_dists.loc[(col,'Missing'),'value'] = f"{n_missing} ({(perc_missing).round(self._decimals)})"

    else:
      raise ValueError("format must be '%', 'N', or 'N (%)'")

    return df_dists
  
  
  # method to add overall counts to the table
  def _add_overall_counts(self,
                           df,
                           df_dists
                           ) -> pd.DataFrame(): # type: ignore

    if self._thousands_sep:
      df_dists.loc[('Overall', ' '), 'value'] = f"{len(df):,}"
    else:
      df_dists.loc[('Overall', ' '), 'value'] = len(df)


    return df_dists
  
  # method to add label_suffix to the table
  def _add_label_suffix(self,
                         col: str,
                         df_dists: pd.DataFrame(),
                         suffix: str,
                         ) -> pd.DataFrame(): # type: ignore

    new_col = col + suffix
    df_dists = df_dists.rename(index={col: new_col}) 

    return df_dists
  
  # method to rename columns
  def _rename_columns(self,
                       df_dists: pd.DataFrame(),
                       col: str,
                      ) -> pd.DataFrame():
    
    return self._rename[col], df_dists.rename(index={col: self._rename[col]})
  
  def _build(self):

    table = pd.DataFrame()

    # get the unique values, before any exclusion, for categorical variables
    original_uniques = self._get_original_uniques(self._categorical)

    for i, df in enumerate(self._dfs):

      df_dists = pd.DataFrame()

      # get distribution for categorical variables
      for col in self._categorical:

        counts = self._my_value_counts(df, original_uniques, col)

        melted_counts = pd.melt(counts.reset_index(), id_vars=['index']) \
                          .set_index(['variable','index'])

        df_dists = pd.concat([df_dists, melted_counts], axis=0)

        if self._missingness:
          df_dists = self._add_missing_counts(df, col, df_dists)

        # rename if applicable
        if col in self._rename.keys():
          col, df_dists = self._rename_columns(df_dists, col)

        if self._label_suffix:
            df_dists = self._add_label_suffix(col, df_dists, ', ' + self._format_cat)
          

      # get distribution for normal variables
      for col in self._normal:
  
          df_dists = self._normal_vars_dist(df, col, df_dists)
          
          if self._missingness:
            df_dists = self._add_missing_counts(df, col, df_dists)

          if col in self._rename.keys():
            col, df_dists = self._rename_columns(df_dists, col)

          if self._label_suffix:
            df_dists = self._add_label_suffix(col, df_dists, ', ' + self._format_normal)
        
      # get distribution for nonnormal variables
      for col in self._nonnormal:

        df_dists = self._nonnormal_vars_dist(df, col, df_dists)

        if self._missingness:
          df_dists = self._add_missing_counts(df, col, df_dists)
        
        if col in self._rename.keys():
          col, df_dists = self._rename_columns(df_dists, col)

        if self._label_suffix:
          df_dists = self._add_label_suffix(col, df_dists, ', ' + self._format_nonnormal)


      df_dists = self._add_overall_counts(df, df_dists)
    
      df_dists.rename(columns={'value': i}, inplace=True)
      table = pd.concat([table, df_dists], axis=1)

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
    


class TableDrifts():
  def __init__(
      self,
      dfs: list,
      categorical: Optional[list] = None,
      normal: Optional[list] = None,
      nonnormal: Optional[list] = None,
      decimals: Optional[int] = 1,
      missingness: Optional[bool] = True,
      rename: Optional[dict] = None,
  ) -> pd.DataFrame:
    
    if not isinstance(dfs, list) or len(dfs) < 1:
        raise ValueError("dfs must be a list with length ≥ 1")
    
    if not isinstance(categorical, list):
        raise ValueError("categorical must be a list")

    if not isinstance(normal, list):
        raise ValueError("normal must be a list")
    
    if not isinstance(nonnormal, list):
        raise ValueError("nonnormal must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    if not isinstance(rename, dict):
        raise ValueError("rename must be a dictionary")
    
    self._dfs = dfs
    self._categorical = categorical
    self._normal = normal
    self._nonnormal = nonnormal
    self._decimals = decimals
    self._missingness = missingness
    self._rename = rename
  
    self._table_flows = TableFlows(
      dfs,
      label_suffix=False,
      thousands_sep=False,
    ).table

    self._table_characteristics = TableCharacteristics(
      dfs,
      categorical=self._categorical,
      normal=self._normal,
      nonnormal=self._nonnormal,
      decimals=self._decimals,
      missingness=False,
      format_cat='N',
      format_normal='Mean',
      format_nonnormal='Mean',
      thousands_sep=False,
      label_suffix=False,
      rename=self._rename,
    ).table

    # auxiliary tables
    self._table_cat_n = TableCharacteristics(
      dfs,
      categorical=self._categorical,
      normal=self._normal,
      nonnormal=self._nonnormal,
      decimals=self._decimals,
      format_cat='N',
      format_normal='Mean',
      format_nonnormal='Mean',
      thousands_sep=False,
      missingness=self._missingness,
      label_suffix=False,
      rename=self._rename,
    ).table

    self._table_cat_perc = TableCharacteristics(
      dfs,
      categorical=self._categorical,
      normal=self._normal,
      nonnormal=self._nonnormal,
      decimals=self._decimals,
      format_cat='%',
      format_normal='SD',
      format_nonnormal='SD',
      thousands_sep=False,
      missingness=self._missingness,
      label_suffix=False,
      rename=self._rename,
    ).table

    self.table = self._build()

    return self.table


  def _build(self):

    inverse_rename = {value: key for key, value in self._rename.items()}

    table = pd.DataFrame(index=self._table_characteristics.index,
                         columns=self._table_flows.columns)
    
    for i, index_name in enumerate(self._table_characteristics.index):
      for j, column_name in enumerate(self._table_flows.columns):
        # skip if index_name is 'Overall' or 'Missing'
        if (index_name[0] == 'Overall'): # | (index_name[1] == 'Missing'):
          table.iloc[i,j] = ''
          continue
        
        # use cat_smd for categorical variables
        if inverse_rename[index_name[0]] in self._categorical:
          cat_n_0 = self._table_cat_n.loc[index_name, :].iloc[j]
          cat_perc_0 = self._table_cat_perc.loc[index_name, :].iloc[j]
          cat_n_1 = self._table_cat_n.loc[index_name, :].iloc[j+1]
          cat_perc_1 = self._table_cat_perc.loc[index_name, :].iloc[j+1]
          table.iloc[i,j] = self._cat_smd(
             prop1=[cat_perc_0/100],
             prop2=[cat_perc_1/100],
             n1=cat_n_0,
             n2=cat_n_1,
             unbiased=True
          )
        
        # use cont_smd for continuous variables
        elif (inverse_rename[index_name[0]] in self._normal) | (inverse_rename[index_name[0]] in self._nonnormal):
          mean_0 = self._table_cat_n.loc[index_name, :].iloc[j]
          sd_0 = self._table_cat_perc.loc[index_name, :].iloc[j]
          mean_1 = self._table_cat_n.loc[index_name, :].iloc[j+1]
          sd_1 = self._table_cat_perc.loc[index_name, :].iloc[j+1]
          n_0 = self._table_characteristics.loc[('Overall', ' '), :].iloc[j]
          n_1 = self._table_characteristics.loc[('Overall', ' '), :].iloc[j+1]
          table.iloc[i,j] = self._cont_smd(
             mean1=mean_0,
             mean2=mean_1,
             sd1=sd_0,
             sd2=sd_1,
             n1=n_0,
             n2=n_1,
             unbiased=True
          )
          
    return table
        

  # adapted from: https://github.com/tompollard/tableone/blob/main/tableone/tableone.py#L659
  def _cat_smd(self,
               prop1=None,
               prop2=None,
               n1=None,
               n2=None,
               unbiased=False):
      """
      Compute the standardized mean difference (regular or unbiased) using
      either raw data or summary measures.

      Parameters
      ----------
      prop1 : list
          Proportions (range 0-1) for each categorical value in dataset 1
          (control). 
      prop2 : list
          Proportions (range 0-1) for each categorical value in dataset 2
          (treatment).
      n1 : int
          Sample size of dataset 1 (control).
      n2 : int
          Sample size of dataset 2 (treatment).
      unbiased : bool
          Return an unbiased estimate using Hedges' correction. Correction
          factor approximated using the formula proposed in Hedges 2011.
          (default = False)

      Returns
      -------
      smd : float
          Estimated standardized mean difference.
      """
      # Categorical SMD Yang & Dalton 2012
      # https://support.sas.com/resources/papers/proceedings12/335-2012.pdf
      prop1 = np.asarray(prop1)
      prop2 = np.asarray(prop2)

      lst_cov = []
      for p in [prop1, prop2]:
          variance = p * (1 - p)
          covariance = - np.outer(p, p)  # type: ignore
          covariance[np.diag_indices_from(covariance)] = variance
          lst_cov.append(covariance)

      mean_diff = np.asarray(prop2 - prop1).reshape((1, -1))  # type: ignore
      mean_cov = (lst_cov[0] + lst_cov[1])/2

      try:
          sq_md = mean_diff @ np.linalg.inv(mean_cov) @ mean_diff.T
      except np.linalg.LinAlgError:
          sq_md = 0

      try:
          smd = np.asarray(np.sqrt(sq_md))[0][0]
      except IndexError:
          smd = 0

      # standard error
      # v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))  # type: ignore
      # se = np.sqrt(v_d)

      if unbiased:
          # Hedges correction (J. Hedges, 1981)
          # Approximation for the the correction factor from:
          # Introduction to Meta-Analysis. Michael Borenstein,
          # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
          # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
          j = 1 - (3/(4*(n1+n2-2)-1))  # type: ignore
          smd = j * smd
          # v_g = (j ** 2) * v_d
          # se = np.sqrt(v_g)

      return np.round(smd, self._decimals)
  
    # adapted from: https://github.com/tompollard/tableone/blob/main/tableone/tableone.py#L581
  def _cont_smd(self,
                mean1=None, mean2=None,
                sd1=None, sd2=None,
                n1=None, n2=None,
                unbiased=False):
    """
    Compute the standardized mean difference (regular or unbiased) using
    either raw data or summary measures.

    Parameters
    ----------
    mean1 : float
        Mean of dataset 1 (control).
    mean2 : float
        Mean of dataset 2 (treatment).
    sd1 : float
        Standard deviation of dataset 1 (control).
    sd2 : float
        Standard deviation of dataset 2 (treatment).
    n1 : int
        Sample size of dataset 1 (control).
    n2 : int
        Sample size of dataset 2 (treatment).
    unbiased : bool
        Return an unbiased estimate using Hedges' correction. Correction
        factor approximated using the formula proposed in Hedges 2011.
        (default = False)

    Returns
    -------
    smd : float
        Estimated standardized mean difference.
    """

    # cohens_d
    denominator = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2) 
    if denominator == 0:
       return 0
    else:
      smd = (mean2 - mean1) / denominator # type: ignore

    # standard error
    # v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))  # type: ignore
    # se = np.sqrt(v_d)

    if unbiased:
        # Hedges correction (J. Hedges, 1981)
        # Approximation for the the correction factor from:
        # Introduction to Meta-Analysis. Michael Borenstein,
        # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
        # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
        j = 1 - (3/(4*(n1+n2-2)-1))  # type: ignore
        smd = j * smd
        # v_g = (j ** 2) * v_d
        # se = np.sqrt(v_g)

    return np.round(smd, self._decimals)
