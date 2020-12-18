```bash
cd pytorch_impl
pip install -r requirements.txt

cd log_uniform
python setup.py install
cd ..
```

--dataset fb để chạy bộ Fb-25k

```
python train_pytorch_SANNE.py --batch_size 64 --num_self_att_layers 2 --num_heads 2 --ff_hidden_size 256 --num_neighbors 4 --walk_length 8 --num_walks 32 --learning_rate 0.0001
```
