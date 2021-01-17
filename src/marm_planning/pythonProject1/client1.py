import socket
import time


def client(joint):
    mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysocket.connect(('127.0.0.1', 7777))
    mysocket.send(bytes(joint, encoding="utf-8"))
    time.sleep(2.0)
    while True:
        data = mysocket.recv(1024)
        datas = str(data, encoding='utf-8')
        if datas:
            return datas
        else:
            break
    mysocket.close()


if "__name__" == "__main__":
    client([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
