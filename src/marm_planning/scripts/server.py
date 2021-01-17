#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket


def server():
    ser = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ser.bind(('127.0.0.1', 7777))
    ser.listen(5)
    while 1:
        client, addr = ser.accept()
        print 'accept %s connect' % (addr,)
        data = client.recv(1024)
        data = str(data)
        print data
        client.send(bytes('get'))
        client.close()


server()
