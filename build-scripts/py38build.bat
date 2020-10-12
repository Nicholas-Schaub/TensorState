call conda create --name py38 python=3.8 -y
call conda activate py38
call pip install numpy cython==3.0a1
call python setup.py sdist bdist_wheel
call conda activate base
call conda remove --name py38 --all -y