#!/bin/bash -x

set -xe
export PYTHONPATH=${WORKSPACE}
export PYTHONIOENCODING=utf-8
export WORKSPACE=${PWD}

cd "${WORKSPACE}"
if [ ! -d "${WORKSPACE}"/env ]; then
  python3 -m venv "${WORKSPACE}"/env
fi
source "${WORKSPACE}/env/bin/activate"

"${VIRTUAL_ENV}"/bin/pip install -U pip setuptools wheel
"${VIRTUAL_ENV}"/bin/pip install -e .

python stock_predictions/main.py -s MSFT -e 5 --v1
