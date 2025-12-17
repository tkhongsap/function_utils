"""General data science utilities."""

from .pandas_helpers import optimize_dataframe
from .statistics import detect_outliers
from .visualization import quick_plot

__all__ = [
    'optimize_dataframe',
    'detect_outliers',
    'quick_plot',
]
