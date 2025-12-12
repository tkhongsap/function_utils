"""
Data Science Utilities - Pandas helpers, visualization, and statistical analysis
"""

from .pandas_helpers import (
    optimize_dataframe,
    chunked_read_csv,
    profile_dataframe,
    memory_usage_summary
)
from .visualization import (
    quick_plot,
    plot_distribution,
    plot_correlation_matrix,
    plot_time_series
)
from .statistics import (
    detect_outliers,
    calculate_correlation,
    distribution_summary,
    test_normality
)
from .gram_schmidt import (
    normalize,
    inner_product,
    orthogonalize,
)

__all__ = [
    # Pandas helpers
    "optimize_dataframe",
    "chunked_read_csv",
    "profile_dataframe",
    "memory_usage_summary",
    # Visualization
    "quick_plot",
    "plot_distribution",
    "plot_correlation_matrix",
    "plot_time_series",
    # Statistics
    "detect_outliers",
    "calculate_correlation",
    "distribution_summary",
    "test_normality",
    # Linear algebra
    "normalize",
    "inner_product",
    "orthogonalize",
]
