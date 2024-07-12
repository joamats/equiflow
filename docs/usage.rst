Usage
=====

.. _installation:

Installation
------------

To use equiflow, first install it using pip:

.. code-block:: console

   $ pip install equiflow

Using equiflow
---------------

To harmonize EHR data and retrieve generic drug names, you can use the ``equiflow`` class and its methods.
Below is an example of how to set up and use EHRmonize:

.. code-block:: python

   from equiflow import EquiFlow
   import pandas as pd

   data = pd.read_csv('data.csv')

   eqfl = EquiFlow(
      data,
      initial_cohort_label = 'in original dataset',
      categorical = ['sex', 'race', 'english'],
      nonnormal = ['sofa']
   )

   eqfl.add_exclusion(
      mask = data.english.notnull(),
      exclusion_reason = 'missing English Proficiency',
      new_cohort_label = 'with English Proficiency data'
   )

   eqfl.plot_flows()