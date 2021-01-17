import datadeal


def Action():
    jointsValue = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    path = "/home/you/jointsState.csv"
    head = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    datadeal.create_csv(path, head)
    datadeal.append_csv(path, jointsValue)


Action()

