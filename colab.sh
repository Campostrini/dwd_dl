source activate py37
cd /content/dwd-dl-thesis
PYTHONPATH=$(pwd)
source activate py37
python dwd_dl/train.py --save true --filename first_run_on_drive