import os
import signal

os.system("killall gzserver")


def kill(pid):
    print('pid', pid)
    os.kill(pid, signal.SIGKILL)


def kill_target(target):
    cmd_run = "ps aux | grep {}".format(target)
    out = os.popen(cmd_run).read()
    pid = int(out.split()[1])
    print(pid)
    kill(pid)


if __name__ == '__main__':
    kill_target('reboot_sim.py')
