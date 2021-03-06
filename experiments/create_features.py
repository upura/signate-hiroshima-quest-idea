import pandas as pd
import pykakasi
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from ayniy.utils import Data


def convert2roman(text):
    kks = pykakasi.kakasi()
    kks.setMode('H', 'a')
    kks.setMode('K', 'a')
    kks.setMode('J', 'a')
    conv = kks.getConverter()
    result = conv.do(text)
    return result


PLAYER_USE_COL = [
    '年度', '選手ID', '投', '打'
]
PITCH_USE_COL = [
    '球種', '投球位置区域', '年度', '試合種別詳細',
    'イニング内打席数', '打席内投球数', '投手ID', '投手投球左右', '投手役割', '投手登板順',
    '投手試合内対戦打者数', '投手試合内投球数', '投手イニング内投球数', '打者ID', '打者打席左右',
    '打者試合内打席数', 'プレイ前ホームチーム得点数', 'プレイ前アウェイチーム得点数',
    'プレイ前アウト数', 'プレイ前ボール数', 'プレイ前ストライク数', 'プレイ前走者状況'
]
DATETIME_COL = []
DELETE_COL = [
    '年度', '投手ID', '打者ID', '選手ID_p', '選手ID_b'
]
CATEGORICAL_COL = [
    '試合種別詳細', '球種', '投球位置区域', '投手投球左右', '投手役割',
    '打者打席左右', '投_p', '打_p',
    'プレイ前走者状況', '投_b', '打_b'
]
fename = 'fe003'


if __name__ == '__main__':
    train_player = pd.read_csv('../input/train_player.csv')
    train_pitch = pd.read_csv('../input/train_pitch.csv')

    train_player = train_player[PLAYER_USE_COL]
    train_pitch = train_pitch[PITCH_USE_COL]
    train_pitch = train_pitch[train_pitch['試合種別詳細'] != 'パ・リーグ公式戦'].reset_index(drop=True)

    # 投手情報の紐付け
    train = pd.merge(train_pitch, train_player,
                     left_on=['年度', '投手ID'], right_on=['年度', '選手ID'],
                     how='inner')

    # 打者情報の紐付け
    train = pd.merge(train, train_player,
                     left_on=['年度', '打者ID'], right_on=['年度', '選手ID'],
                     how='inner', suffixes=('_p', '_b'))

    train.drop(DELETE_COL, axis=1, inplace=True)

    for f in CATEGORICAL_COL:
        lbl = preprocessing.LabelEncoder()
        train[f] = lbl.fit_transform(list(train[f].values))

    print(train.columns)
    cname = pd.DataFrame()
    cname['before'] = train.columns
    train.columns = [convert2roman(d) for d in train.columns]
    cname['after'] = train.columns
    cname.to_csv('../input/cname.csv', index=False)

    print('train.shape: ', train.shape)
    print(train.head())
    print([convert2roman(d) for d in CATEGORICAL_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        train.drop('shiaishubetsushousai', axis=1),
        train['shiaishubetsushousai'], test_size=0.2,
        stratify=train['shiaishubetsushousai'],
        random_state=42)

    Data.dump(X_train, f'../input/X_train_{fename}.pkl')
    Data.dump(y_train, f'../input/y_train_{fename}.pkl')
    Data.dump(X_test, f'../input/X_test_{fename}.pkl')
    Data.dump(y_test, f'../input/y_test_{fename}.pkl')
