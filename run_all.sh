NUM_WALKS=$1
PATH_LENGTH=$2

DIR_NAME="walks_${NUM_WALKS}_path_${PATH_LENGTH}"
mkdir embeddings/${DIR_NAME}

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_uniform --bias-val 1000 --walk-selection uniformly_random

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_self_avoiding --bias-val 1000 --walk-selection self_avoiding

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_nobacktracking --bias-val 1000 --walk-selection no_backtracking

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_mbrw2 --bias-val 2 --walk-selection mbrw

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_mbrw10 --bias-val 10 --walk-selection mbrw

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_mbrw100 --bias-val 100 --walk-selection mbrw

python ./intelliJDebug.py --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks $NUM_WALKS --representation-size 128 --walk-length $PATH_LENGTH --window-size 10 --workers 8 --output embeddings/${DIR_NAME}/blogcatalog_mbrw1000 --bias-val 1000 --walk-selection mbrw