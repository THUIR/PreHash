# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
import json

np.random.seed(DEFAULT_SEED)
host_name = socket.gethostname()
print(host_name)

RAW_DATA = '/path/to/raw/data/'  # http://jmcauley.ucsd.edu/data/amazon/


# hc691做好的amazon数据集，给id + 1
def format_hc691_rating(in_csv, out_csv):
    in_df = pd.read_csv(in_csv, header=None, names=[UID, IID])
    if in_df[UID].min() == 0:
        in_df[UID] += 1
    if in_df[IID].min() == 0:
        in_df[IID] += 1
    in_df.to_csv(out_csv, sep='\t', index=False)
    return in_df


# http://jmcauley.ucsd.edu/data/amazon/
# format amazon ratings only 数据集
def format_rating_only(in_csv, out_csv, label01=False):
    in_df = pd.read_csv(in_csv, header=None, names=[UID, IID, LABEL, TIME])
    # 按时间、uid、iid排序
    out_df = in_df.sort_values(by=[TIME, UID, IID]).reset_index(drop=True)

    # 给uid编号，从1开始
    uids = sorted(out_df[UID].unique())
    uid_dict = dict(zip(uids, range(1, len(uids) + 1)))
    out_df[UID] = out_df[UID].apply(lambda x: uid_dict[x])

    # 给iid编号，从1开始
    iids = sorted(out_df[IID].unique())
    iid_dict = dict(zip(iids, range(1, len(iids) + 1)))
    out_df[IID] = out_df[IID].apply(lambda x: iid_dict[x])

    # # 丢掉时间戳
    # out_df = out_df.drop(columns=TIME)

    # 如果要format成0（负向）- 1（正向）两种label，而不是评分，则认为评分大于3的表示喜欢为1，否则不喜欢为0
    if label01:
        out_df[LABEL] = out_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', out_df[LABEL].min(), out_df[LABEL].max())
    print(Counter(out_df[LABEL]))

    out_df.to_csv(out_csv, sep='\t', index=False)
    print(out_df)
    return out_df


# http://jmcauley.ucsd.edu/data/amazon/
# format amazon 5-core 数据集
def format_5core(in_json, out_csv, label01=False):
    # 读入json文件
    records = []
    for line in open(in_json, 'r'):
        record = json.loads(line)
        records.append(record)
    # 讲json信息转换问pandas DataFrame
    out_df = pd.DataFrame()
    out_df[UID] = [r['reviewerID'] for r in records]
    out_df[IID] = [r['asin'] for r in records]
    out_df[LABEL] = [r['overall'] for r in records]
    out_df[TIME] = [r['unixReviewTime'] for r in records]

    # 按时间、uid、iid排序
    out_df = out_df.sort_values(by=[TIME, UID, IID])
    out_df = out_df.drop_duplicates([UID, IID]).reset_index(drop=True)

    # 给uid编号，从1开始
    uids = sorted(out_df[UID].unique())
    uid_dict = dict(zip(uids, range(1, len(uids) + 1)))
    out_df[UID] = out_df[UID].apply(lambda x: uid_dict[x])

    # 给iid编号，从1开始
    iids = sorted(out_df[IID].unique())
    iid_dict = dict(zip(iids, range(1, len(iids) + 1)))
    out_df[IID] = out_df[IID].apply(lambda x: iid_dict[x])

    # # 丢掉时间戳
    # out_df = out_df.drop(columns=TIME)

    # 如果要format成0（负向）- 1（正向）两种label，而不是评分，则认为评分大于3的表示喜欢为1，否则不喜欢为0
    if label01:
        out_df[LABEL] = out_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', out_df[LABEL].min(), out_df[LABEL].max())
    print(Counter(out_df[LABEL]))

    out_df.to_csv(out_csv, sep='\t', index=False)
    # print(out_df)
    return out_df


def main():
    all_data_file = os.path.join(DATA_DIR, 'ratings_Books.csv')
    format_rating_only(in_csv=os.path.join(RAW_DATA, 'ratings_Books.csv'),
                       out_csv=all_data_file, label01=False)
    dataset_name = 'Books-1-1'
    leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=1)

    all_data_file = os.path.join(DATA_DIR, 'ratings_Grocery_and_Gourmet_Food.csv')
    format_rating_only(in_csv=os.path.join(RAW_DATA, 'ratings_Grocery_and_Gourmet_Food.csv'),
                       out_csv=all_data_file, label01=False)
    dataset_name = 'Grocery-1-1'
    leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=1)

    all_data_file = os.path.join(DATA_DIR, 'ratings_Pet_Supplies.csv')
    format_rating_only(in_csv=os.path.join(RAW_DATA, 'ratings_Pet_Supplies.csv'),
                       out_csv=all_data_file)
    dataset_name = 'Pet-1-1'
    leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=1)

    all_data_file = os.path.join(DATA_DIR, 'ratings_Video_Games.csv')
    format_rating_only(in_csv=os.path.join(RAW_DATA, 'ratings_Video_Games.csv'),
                       out_csv=all_data_file)
    dataset_name = 'VideoGames-1-1'
    leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=1)
    return


if __name__ == '__main__':
    main()
