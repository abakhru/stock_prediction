#!/usr/bin/env python

# Trying PyCaret experiments
"""
- https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e
- brew install libomp
"""

from pycaret.classification import *
from pycaret.datasets import get_data


# setup data
index = get_data('index')

data = get_data('juice')

# Initialize setup (when using Notebook environment)
# clf1 = setup(data, target = 'SalePrice')

# Initialize setup (outside of Notebook environment)
# clf1 = setup(data, target = 'target-variable', html = False)

# Initialize setup (When using remote execution such as Kaggle / GitHub actions / CI-CD pipelines)
# clf1 = setup(data, target = 'target-variable', html = False, silent = True)
clf1 = setup(data, target = 'Purchase', session_id=123, log_experiment=False, experiment_name='bank1')

# return best model
best = compare_models()

