import sys
sys.path.insert(0, '.\\ID3')

import pandas as pd
from math import log2
from node import Node


class ID3tree:
    def __init__(self, data: pd.DataFrame,
                 y: pd.Series,
                 window: float):
        self.attr_encode = self.encode(data)
        # self.win_data = data.sample(frac=window)
        self.y_name = y.name
        self.y = y
        self.data = pd.concat([data, y], axis=1)
        self.root = Node()

    def encode(self, win_dataf):
        encodes = {}
        attr = win_dataf.columns

        for i in range(len(attr)):
            attr_choose = attr[i]
            attr_info = win_dataf.value_counts(attr_choose)
            encodes[attr_choose] = {j: attr_info.index[j]
                                    for j in range(len(attr_info))}
        return encodes

    def info(self, dataf: pd.DataFrame,
             attr: str,
             ylabel: str):
        res = 0
        num = len(dataf)

        for label, grouped in dataf.groupby(attr):
            value_series = grouped.value_counts(ylabel).reset_index(drop=True)
            num_piece = len(grouped)
            info_piece = [-value_series[i]*log2(value_series[i]/num_piece)/num_piece
                          for i in range(len(value_series))]
            print(info_piece)
            res = res + num_piece * sum(info_piece) / num

        return res

    def info_e(self, dataf: pd.DataFrame,
               attr: str,
               ylabel: str):
        info = 0
        num = len(dataf)
        y_info = dataf.value_counts(ylabel).reset_index(drop=True)
        for i in range(len(y_info)):
            info = info - y_info[i] * log2(y_info[i] / num) / num
        info_condition = self.info(dataf, attr, ylabel)
        info_sum = sum([-(len(grouped)*log2(len(grouped)/num))/num
                        for _, grouped in dataf.groupby(attr)])
        res = (info - info_condition) / info_sum
        return res

    def select_attr(self, dataf: pd.DataFrame, ylabel):
        """
        first we construct the tree as large as we can, then we prune it upward.
        """
        attr_list = dataf.columns
        info_list = []

        # return if all attrs have the same value -- we cannot split anymore
        if len(dataf.value_counts(list(set(list(attr_list)) - set([ylabel])))) == 1:
            return False, ''

        for i in range(len(attr_list)):         # compute conditional info
            if attr_list[i] == ylabel:
                info_list.append(0)
                continue
            info_list.append(self.info(dataf, attr_list[i], ylabel))

        print('test: ', self.info(dataf, 'Pclass', ylabel))

        print(info_list)
        print(dataf.head(10))

        info_mean = sum(info_list) / (len(attr_list) - 1)
        # if conditional info is smaller than the mean, we compute info gain ratio
        for i in range(len(info_list)):
            if info_list[i] < info_mean and attr_list[i] != ylabel:
                info_list[i] = self.info_e(dataf, attr_list[i], ylabel)
            else:
                info_list[i] = 0

        max_value = max(info_list)
        attr = attr_list[info_list.index(max_value)]
        return True, attr

    def gettree(self, dataf: pd.DataFrame, ylabel: str):
        condition1 = dataf.value_counts(ylabel)
        if len(condition1) == 1:
            return Node(res=condition1.index[0])

        flag, attr_selected = self.select_attr(dataf, ylabel)
        if not flag:
            return Node(res=dataf.mode(axis=0).loc[:, ylabel][0])
        node = Node(attr=attr_selected, res=dataf.mode(axis=0).loc[:, ylabel][0])

        print(attr_selected)
        for label, group in dataf.groupby(attr_selected):
            node.fork_dict[label] = self.gettree(group, ylabel)

        print(len(dataf))

        return node

    def fit(self):
        self.root = self.gettree(dataf=self.data, ylabel=self.y_name)
        # self.print_tree(self.root)
        return

    @staticmethod
    def data_loader(data: pd.DataFrame):
        data_length = data.shape[0]
        data.reindex(index=range(data_length))
        for i in range(data_length):
            yield data.iloc[i]
        return

    def classify(self, x: pd.Series):
        state = self.root
        while len(state.fork_dict) != 0 and state is not None:
            attr_select = state.trial
            attr_value = x.loc[attr_select]
            if attr_value not in state.fork_dict.keys():
                return state.res
            state = state.fork_dict[attr_value]
        if state is None:
            print("ERROR: state is None")
            sys.exit()
        return state.res

    def predict(self, data: pd.DataFrame):
        """
        do classification with x, it will return an n-dimensions array
        """
        loader = self.data_loader(data)
        x = list(loader)
        res = [[self.classify(i)] for i in x]
        return res

    def print_tree(self, node):
        if len(node.fork_dict) == 0:
            print('res: ', node.res)
            return

        print(node.trial)
        nodes_next = list(node.fork_dict.values())
        for i in range(len(nodes_next)):
            self.print_tree(nodes_next[i])
        print('-------------------------')
        return
