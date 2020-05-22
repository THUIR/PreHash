# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
from utils import utils
import pandas as pd
from collections import Counter

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '/path/to/raw/data/'  # http://www.recsyschallenge.com/2017/

USERS_FILE = os.path.join(RAW_DATA, 'users.csv')
ITEMS_FILE = os.path.join(RAW_DATA, 'items.csv')
INTERACTIONS_FILE = os.path.join(RAW_DATA, 'interactions.csv')

RECSYS2017_DATA_DIR = os.path.join(DATA_DIR, 'RecSys2017')
USER_FEATURE_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017' + USER_SUFFIX)
ITEM_FEATURE_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017' + ITEM_SUFFIX)
ALL_DATA_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017.all.csv')


def format_data(sample_neg=-1.0):
    user_df = pd.read_csv(USERS_FILE, sep='\t')
    user_columns = ['user_id', 'u_jobroles', 'u_career_level', 'u_discipline_id', 'u_industry_id',
                    'u_country', 'u_region', 'u_experience_n_entries_class', 'u_experience_years_experience',
                    'u_experience_years_in_current',
                    'u_edu_degree', 'u_edu_fieldofstudies', 'u_wtcj', 'u_premium']
    user_df.columns = user_columns
    user_df = user_df.drop_duplicates('user_id')
    u_drop_columns = ['u_jobroles', 'u_edu_fieldofstudies']
    user_df = user_df.drop(columns=u_drop_columns)
    user_df = user_df.sort_values('user_id').reset_index(drop=True)
    country_dict = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    user_df['u_country'] = user_df['u_country'].apply(lambda x: country_dict[x])
    uid_list = user_df['user_id'].unique()
    uid_dict = dict(zip(uid_list, range(1, len(uid_list) + 1)))
    user_df[UID] = user_df['user_id'].apply(lambda x: uid_dict[x])
    user_df = user_df[[UID] + [c for c in user_columns if c not in u_drop_columns + ['user_id']]]

    item_df = pd.read_csv(ITEMS_FILE, sep='\t')
    item_columns = ['item_id', 'i_title', 'i_career_level', 'i_discipline_id', 'i_industry_id',
                    'i_country', 'i_is_paid', 'i_region', 'i_latitude', 'i_longitude',
                    'i_employment', 'i_tags', 'i_created_at']
    item_df.columns = item_columns
    item_df = item_df.drop_duplicates('item_id')
    i_drop_columns = ['i_title', 'i_tags']
    item_df = item_df.drop(columns=i_drop_columns)
    item_df = item_df.sort_values('item_id').reset_index(drop=True)
    item_df['i_country'] = item_df['i_country'].apply(lambda x: country_dict[x])
    item_df['i_latitude'] = item_df['i_latitude'].apply(lambda x: 0 if np.isnan(x) else int((int(x + 90) / 10)) + 1)
    item_df['i_longitude'] = item_df['i_longitude'].apply(lambda x: 0 if np.isnan(x) else int((int(x + 180) / 10)) + 1)
    item_df['i_created_at'] = pd.to_datetime(item_df['i_created_at'], unit='s')
    item_year = item_df['i_created_at'].apply(lambda x: x.year)
    min_year = item_year.min()
    item_month = item_df['i_created_at'].apply(lambda x: x.month)
    item_df['i_created_at'] = (item_year.fillna(-1) - min_year) * 12 + item_month.fillna(-1)
    item_df['i_created_at'] = item_df['i_created_at'].apply(lambda x: int(x) if x > 0 else 0)
    iid_list = item_df['item_id'].unique()
    iid_dict = dict(zip(iid_list, range(1, len(iid_list) + 1)))
    item_df[IID] = item_df['item_id'].apply(lambda x: iid_dict[x])
    item_df = item_df[[IID] + [c for c in item_columns if c not in i_drop_columns + ['item_id']]]

    inter_df = pd.read_csv(INTERACTIONS_FILE, sep='\t')
    inter_columns = ['user_id', 'item_id', LABEL, TIME]
    inter_df.columns = inter_columns
    inter_df = inter_df.dropna().astype(int)
    inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 0 if x == 4 or x == 0 else 1)
    inter_df = inter_df.sort_values(by=LABEL).reset_index(drop=True)
    print(inter_df)
    inter_df = inter_df.drop_duplicates(['user_id', 'item_id'], keep='last')
    print(inter_df[LABEL].value_counts())

    pos_df = inter_df[inter_df[LABEL] > 0]
    if sample_neg < 0:
        neg_df = inter_df[inter_df[LABEL] == 0].sample(n=len(pos_df), replace=False)
        inter_df = pd.concat([pos_df, neg_df])
    elif sample_neg == 0:
        inter_df = pos_df
    elif 0 < sample_neg < 1:
        neg_df = inter_df[inter_df[LABEL] == 0].sample(frac=sample_neg, replace=False)
        inter_df = pd.concat([pos_df, neg_df])
    inter_df = inter_df.sort_values(by=TIME).reset_index(drop=True)
    inter_df[UID] = inter_df['user_id'].apply(lambda x: uid_dict[x])
    inter_df[IID] = inter_df['item_id'].apply(lambda x: iid_dict[x])
    inter_df = inter_df[[UID, IID, LABEL, TIME]]
    user_df.to_csv(USER_FEATURE_FILE, sep='\t', index=False)
    item_df.to_csv(ITEM_FEATURE_FILE, sep='\t', index=False)
    inter_df.to_csv(ALL_DATA_FILE, sep='\t', index=False)
    return


def main():
    utils.check_dir_and_mkdir(RECSYS2017_DATA_DIR)
    format_data(sample_neg=1.0)

    dataset_name = 'RecSys2017-1-1'
    leave_out_by_time_csv(all_data_file=ALL_DATA_FILE, dataset_name=dataset_name, leave_n=1, warm_n=1,
                          u_f=USER_FEATURE_FILE, i_f=ITEM_FEATURE_FILE)
    return


if __name__ == '__main__':
    main()
