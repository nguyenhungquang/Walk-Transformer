```bash
cd pytorch_impl
pip install -r requirements.txt

cd log_uniform
python setup.py install
```


Dữ liệu: Em lấy 1 bộ dữ liệu freebase15k trong 1 bài out-of-knowledge, lưu trong folder fb15k. Hiện tại train và test được thực hiện trên cùng 1 tập dữ liệu là file train. Mỗi hàng tương ứng với 1 triple (h,r,t)
- File generate_random_walks.py: đọc đồ thị từ dữ liệu và sinh ra các random walks.
- File train_pytorch_SANNE.py đọc dữ liệu, train và evaluate.


```
python train_pytorch_SANNE.py --batch_size 64 --num_self_att_layers 2 --num_heads 2 --ff_hidden_size 256 --num_neighbors 4 --walk_length 8 --num_walks 32 --learning_rate 0.005
```
