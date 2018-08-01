""" A script to hold a namedtuple object for an individual dataframe and its
metadata. This metadata is simply a list of the distributions from which each
column was generated. These are reused during mutation and for filling in
missing values during crossover. """

from collections import namedtuple

Individual = namedtuple("Individual", ["column_metadata", "dataframe"])
