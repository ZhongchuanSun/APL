import time
import numpy as np
import concurrent.futures
from collections import defaultdict


def timer(func):
    """The timer decorator
    """

    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result

    return inner


def load_data(data_file):
    user_pos = defaultdict(list)
    with open(data_file)as fin:
        for line in fin:
            line = line.strip().split()
            user_pos[int(line[0])].append(int(line[1]))
    return user_pos


def get_batch_data(data, index, size):
    column_1 = []
    column_2 = []
    for i in range(index, index + size):
        line = data[i]
        column_1.append(int(line[0]))
        column_2.append(int(line[1]))
    return np.array(column_1), np.array(column_2)


_model = None
_all_items = None
_user_pos_train = None
_user_pos_test = None


@timer
def evaluate_model(model, all_items, user_pos_train, user_pos_test):
    """evaluate model
    Returns:
        [p_10, recall_10, map_10, ndcg_10]
    """
    global _model
    global _all_items
    global _user_pos_train
    global _user_pos_test
    _model = model
    _all_items = set(all_items)
    _user_pos_train = user_pos_train
    _user_pos_test = user_pos_test
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        batch_result = executor.map(eval_one_user, list(test_users))

    result = np.array([0.] * 4)
    for re in batch_result:
        result += re
    ret = result / test_user_num
    ret = list(ret)

    return ret


def eval_one_user(user):
    rating = _model.predict(user)
    unrated = np.array(list(_all_items - set(_user_pos_train[user])), dtype=int)
    ground_truth = _user_pos_test[user]

    unrated_score = rating[unrated]
    index = np.argsort(-unrated_score)
    rank = unrated[index]
    result = []

    result.append(precision_at_k(rank, ground_truth, 10))
    result.append(recall_at_k(rank, ground_truth, 10))
    result.append(ap(rank, ground_truth, 10))
    result.append(ndcg_at_k(rank, ground_truth, 10))

    return np.array(result)


def ndcg_at_k(rank, ground_truth, k):
    """NDCG
    """
    i_dcg = np.sum(1.0 / np.log2(np.arange(2, k + 2)))
    if not i_dcg:
        return 0.
    dcg = np.sum([1.0 / np.log2(i + 2) if rank[i] in ground_truth else 0.0 for i in range(len(rank[:k]))])
    return dcg / i_dcg


def precision_at_k(rank, ground_truth, k):
    """Precision
    """
    return 1.0 * len(set(rank[:k]) & set(ground_truth)) / k


def recall_at_k(rank, ground_truth, k):
    """Recall
    """
    return 1.0 * len(set(rank[:k]) & set(ground_truth)) / len(ground_truth)


def ap(rank, ground_truth, k):
    """Mean Average Precision
    """
    pre_sum = 0.0
    count = 0
    for i in range(len(rank[:k])):
        if rank[i] in ground_truth:
            count += 1
            pre_sum += count / (i + 1)
    return pre_sum / min(len(ground_truth), k)
