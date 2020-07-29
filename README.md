# PreHash

This is our implementation for the paper:

*Shaoyun Shi, Weizhi Ma, Min Zhang, Yongfeng Zhang, Xinxing Yu, Houzhi Shan, Yiqun Liu, and Shaoping Ma. 2020. [Beyond User Embedding Matrix: Learning to Hash for Modeling Large-Scale Users in Recommendation](https://dl.acm.org/doi/10.1145/3397271.3401119) 
In SIGIR'20.*

**Please cite our paper if you use our codes. Thanks!**

Author: Shaoyun Shi (shisy13 AT gmail.com)

```
@inproceedings{shi2020prehash,
  title={Beyond User Embedding Matrix: Learning to Hash for Modeling Large-Scale Users in Recommendation},
  author={Shaoyun Shi, Weizhi Ma, Min Zhang, Yongfeng Zhang, Xinxing Yu, Houzhi Shan, Yiqun Liu, and Shaoping Ma},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020},
  page={319--328},
  organization={ACM}
}
```



## Environments

Python 3.7.6

Packages: See in [requirements.txt](https://github.com/THUIR/PreHash/blob/master/requirements.txt)

```
pathos==0.2.5
tqdm==4.42.1
numpy==1.18.1
torch==1.1.0
pandas==1.0.1
scikit_learn==0.23.1
```



## Datasets

The processed datasets can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/43991418f4764e2ca6d6/) or [Google Drive](https://drive.google.com/open?id=1uzKQ6yt3wawyoqVG6pElHOvp1AAUTsoa).

You should place the datasets in the `./dataset/`. The tree structure of directories should look like:

```
.
├── dataset
│   ├── Books-1-1
│   ├── Grocery-1-1
│   ├── Pet-1-1
│   ├── RecSys2017-1-1
│   └── VideoGames-1-1
└── src
    ├── data_loaders
    ├── data_processors
    ├── datasets
    ├── models
    ├── runners
    └── utils
```

- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 

- **RecSys2017 Dataset**: The origin dataset can be found [here](http://www.recsyschallenge.com/2017/). 

- The codes for processing the data can be found in [`./src/datasets/`](https://github.com/THUIR/PreHash/tree/master/src/datasets)

    

## Example to run the codes

-   Some running commands can be found in [`./command/command.py`](https://github.com/THUIR/PreHash/blob/master/command/command.py)
-   For example:

```
# PreHash enhanced BiasedMF on Grocery dataset
> cd PreHash/src/
> python main.py --model_name PreHash --dataset Grocery-1-1 --rank 1 --metrics ndcg@10,precision@1 --lr 0.001 --l2 1e-7 --train_sample_n 1 --hash_u_num 1024 --sparse_his 0 --max_his 10 --sup_his 1 --random_seed 2018 --gpu 0
```

