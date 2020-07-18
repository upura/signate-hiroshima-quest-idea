import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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
    X_train['Probability of regular season'] = oof
    X_train = X_train.query('打者チームID==6 and 位置_b!="投手"')
    X_train['uid'] = X_train['試合ID'].astype(str) + '-' + X_train['イニング'].astype(str) + '-' + X_train['イニング内打席数'].astype(str)

    USE_COL = [
        'uid',
        '選手名_b',
        'プレイ前アウト数',
        'Probability of regular season'
    ]

    X_train = X_train[USE_COL].sort_index()
    X_box = X_train.groupby('uid').last().reset_index()

    X_box['出塁率'] = (X_box['プレイ前アウト数'] == X_box['プレイ前アウト数'].shift(-1))
    X_box['調整済み出塁率'] = X_box['出塁率'] * (1 - X_box['Probability of regular season'])
    X_player = X_box.groupby('選手名_b').mean()[['出塁率', '調整済み出塁率']].sort_values('出塁率').reset_index()
    X_player.index = X_player['選手名_b']

    japanize_matplotlib.japanize()
    X_player.plot.barh()
    plt.xlabel('2017年のレギュラーシーズンの出塁率')
    plt.savefig('../output/pred/adjusted.png')
