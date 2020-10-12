call conda create --name py35 python=3.5 -y
call conda activate py35
call pip install numpy cython==3.0a1
call python ../setup.py sdist bdist_wheel
call conda activate base
call conda remove --name py35 --all -y