#!/bin/bash -x

set -xe
export WORKSPACE=${WORKSPACE:-$PWD}
export PYTHONPATH=${WORKSPACE}
export PYTHONIOENCODING=utf-8

cd "${WORKSPACE}"
if [ ! -d "${WORKSPACE}"/env ]; then

  python3.6 -m venv "${WORKSPACE}"/env
fi
source "${WORKSPACE}/env/bin/activate"

"${VIRTUAL_ENV}"/bin/pip install -U pip setuptools wheel
"${VIRTUAL_ENV}"/bin/pip install -e .

brew install libomp
