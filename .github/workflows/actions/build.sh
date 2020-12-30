#!/bin/sh

yum -y install gsl-devel

PYTHON_VERSIONS=("cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    /opt/python/${PYTHON_VERSION}/bin/pip install --upgrade pip
    /opt/python/${PYTHON_VERSION}/bin/pip install -U wheel auditwheel
    /opt/python/${PYTHON_VERSION}/bin/pip install numpy==1.19.2 cython==3.0a1
    /opt/python/${PYTHON_VERSION}/bin/python setup.py bdist_wheel -d dist
done

for whl in ./dist/*.whl; do
    auditwheel repair $whl -w ./dist
    rm $whl
done