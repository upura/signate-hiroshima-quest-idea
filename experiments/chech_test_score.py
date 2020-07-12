from sklearn.metrics import roc_auc_score, confusion_matrix

from ayniy.utils import Data


if __name__ == '__main__':
    pred = Data.load('../output/pred/run000-test.pkl')
    y_test = Data.load('../input/y_test_fe000.pkl')
    print(roc_auc_score(y_test, pred))
    print(confusion_matrix(y_test, (pred > 0.5).astype(int)))
