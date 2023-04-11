class Node:
    def __init__(self, attr='',
                 fork_dict={},
                 res=""):
        self.trial = attr
        self.fork_dict = dict()
        self.res = res


if __name__ == '__main__':
    print("id of node: ")
    print(id(node1), "    ", id(node0))

    print("id of variate: ")
    print(id(node1.fork_dict), "    ", id(node0.fork_dict))

    a = []
    b = a.copy()
    a.append(5)
    print(a, b)
