#!/bin/sh

yum -y install gsl-devel

PYTHON_VERSIONS=("cp38-cp38" "cp39-cp39" "cp310-cp310")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    /opt/python/${PYTHON_VERSION}/bin/pip install --upgrade pip
    /opt/python/${PYTHON_VERSION}/bin/pip install -U wheel auditwheel poetry
    /opt/python/${PYTHON_VERSION}/bin/python -m poetry build
done

for whl in ./dist/*.whl; do
    auditwheel repair $whl -w ./dist
    rm $whl
done
