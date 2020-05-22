# coding=utf-8

# # Grocery-1-1
'python main.py --model_name BiasedMF --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-7 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name NeuMF --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name SVDPP --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-5 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name FISM --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashNeuMF --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHash --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-7 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'

# # Pet-1-1
'python main.py --model_name BiasedMF --dataset Pet-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-7 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name NeuMF --dataset Pet-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name SVDPP --dataset Pet-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name FISM --dataset Pet-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashNeuMF --dataset Pet-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHash --dataset Pet-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-7 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'

# # VideoGames-1-1
'python main.py --model_name BiasedMF --dataset VideoGames-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-7 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name NeuMF --dataset VideoGames-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name SVDPP --dataset VideoGames-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name FISM --dataset VideoGames-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashNeuMF --dataset VideoGames-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHash --dataset VideoGames-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-7 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'

# # Books-1-1
'python main.py --model_name FISM --dataset Books-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-7 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashNeuMF --dataset Books-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-7 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHash --dataset Books-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-7 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'

# # RecSys2017-1-1
'python main.py --model_name SVDPP --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name FISM --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name NeuMF --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashNeuMF --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name BiasedMF --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHash --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name WideDeep --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashWideDeep --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
'python main.py --model_name ACCM --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1  --lr 0.001 --l2 1e-6 --train_sample_n 1 --random_seed 2018 --gpu 0'
'python main.py --model_name PreHashACCM --dataset RecSys2017-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-6 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0'
