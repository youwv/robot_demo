import datadeal


def Pose():
    path = "/home/you/pose.csv"
    head = ["posx", "posy", "posz", "orientationx", "orientationy", "orientationz", "orientationw"]
    datadeal.create_csv(path, head)


Pose()
