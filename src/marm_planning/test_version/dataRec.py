import datadeal


def record():
    path = "/home/you/record.csv"
    head = ["a_temp", "r", "done"]
    datadeal.create_csv(path, head)


record()
