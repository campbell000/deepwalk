import sys
import os
import runpy
path = os.path.dirname(sys.modules[__name__].__file__)
path = os.path.join(path, '..')
sys.path.insert(0, path)
runpy.run_module('deepwalk', run_name="__main__",alter_sys=True)

#
# Do walks
# --format mat --input example_graphs/blogcatalog.mat --max-memory-data-size 0 --number-walks 80 --representation-size 128 --walk-length 40 --window-size 10 --workers 8 --output embeddings/blog_uniformly_random --bias-val 2 --walk-selection self_avoiding
#
# Do ScoringM
# python example_graphs/scoring.py --emb example_embeddings/blog_uniformly_random.embeddings --network example_graphs/blogcatalog.mat --num-shuffle 10 --all