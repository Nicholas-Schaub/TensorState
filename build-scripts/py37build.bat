call conda create --name py37 python=3.7 -y
call conda activate py37
call pip install numpy cython==3.0a1
call cd ..
call python setup.py sdist bdist_wheel
call cd build-scripts
call conda activate base
call conda remove --name py37 --all -y