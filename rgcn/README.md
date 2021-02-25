`pip install dgl
pip install dgl-cu+CUDA-version (i.e. dgl-cu101)`

`python link.py --gpu 0 --max-subgraph-size 1000 --evaluate-every 100 --graph-batch-size 2000 --eval-batch-size 1 --n-epochs 50000 --lr 0.0001 `
- max_subgraph_size: số node ước lượng giới hạn trong subgraph, để padding
- evaluate_every: evaluate sau mỗi k minibatch
- n_epochs: số minibatch
- lr: learning rate
