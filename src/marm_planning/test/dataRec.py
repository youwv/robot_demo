import datadeal


def record():
    path = "/home/you/record.csv"
    head = ["s", "a", "r", "t", "s2"]
    datadeal.create_csv(path, head)


record()
