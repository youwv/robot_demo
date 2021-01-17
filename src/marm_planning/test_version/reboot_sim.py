import os


def reboot():
    os.system("cd ~/robotiq/src/abb_irb1200_gazebo/launch/ ; roslaunch abb_world.launch ")


if __name__ == '__main__':
    reboot()
