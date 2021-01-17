import socket


def client(joint):
    mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysocket.connect(('127.0.0.1', 7777))
    mysocket.send(bytes(joint, encoding="utf-8"))
    while True:
        data = mysocket.recv(1024)
        data = str(data, encoding='utf-8')
        if data:
            print(data)
        else:
            break
    mysocket.close()
