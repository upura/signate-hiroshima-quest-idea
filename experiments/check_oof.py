import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from ayniy.utils import Data


if __name__ == '__main__':
    oof = Data.load('../output/pred/run003-train.pkl')

    train_player = pd.read_csv('../input/train_player.csv')
    train_pitch = pd.read_csv('../input/train_pitch.csv')
    train_pitch = train_pitch[train_pitch['試合種別詳細'] != 'パ・リーグ公式戦'].reset_index(drop=True)

    # 投手情報の紐付け
    train = pd.merge(train_pitch, train_player,
                     left_on=['年度', '投手ID'], right_on=['年度', '選手ID'],
                     how='inner')

    # 打者情報の紐付け
    train = pd.merge(train, train_player,
                     left_on=['年度', '打者ID'], right_on=['年度', '選手ID'],
                     how='inner', suffixes=('_p', '_b'))

    X_train, _, _, _ = train_test_split(
        train.drop('試合種別詳細', axis=1),
        train['試合種別詳細'], test_size=0.2,
        stratify=train['試合種別詳細'],
        random_state=42)
    X_train['Probability of regular season (打者)'] = oof

    df_b = X_train.query('打者チームID==6 and 位置_b!="投手"')
    agg_dict = df_b.groupby('選手名_b').median()['Probability of regular season (打者)'].sort_values().to_dict()
    japanize_matplotlib.japanize()
    aggs = df_b.groupby('選手名_b').median().sort_values(by="Probability of regular season (打者)", ascending=False)
    cols = aggs[:100].index
    best_features = df_b.loc[df_b.選手名_b.isin(cols)]
    best_features['sort_v'] = best_features['選手名_b'].map(agg_dict)
    plt.figure(figsize=(5, 7))
    sns.barplot(x="Probability of regular season (打者)", y="選手名_b", data=best_features.sort_values(by="sort_v", ascending=False))
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('../output/pred/b_result.png')

    X_train['Probability of regular season (投手)'] = oof

    df_p = X_train.query('投手チームID==6')
    agg_dict = df_p.groupby('選手名_p').median()['Probability of regular season (投手)'].sort_values().to_dict()
    japanize_matplotlib.japanize()
    aggs = df_p.groupby('選手名_p').median().sort_values(by="Probability of regular season (投手)", ascending=False)
    cols = aggs[:100].index
    best_features = df_p.loc[df_p.選手名_p.isin(cols)]
    best_features['sort_v'] = best_features['選手名_p'].map(agg_dict)
    plt.figure(figsize=(5, 7))
    sns.barplot(x="Probability of regular season (投手)", y="選手名_p", data=best_features.sort_values(by="sort_v", ascending=False))
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('../output/pred/p_result.png')
