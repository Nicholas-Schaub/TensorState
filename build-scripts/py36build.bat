call conda create --name py36 python=3.6 -y
call conda activate py36
call pip install numpy cython==3.0a1
call python setup.py sdist bdist_wheel
call conda activate base
call conda remove --name py36 --all -y