#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv


def create_csv(path, head):
    with open(path, "w+") as file:
        csv_file = csv.writer(file)
        head = head
        csv_file.writerow(head)


def append_csv(path, transition):
    with open(path, "a+") as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        datas = transition
        csv_file.writerows(datas)


def read_csv(path):
    with open(path, "r+") as file:
        csv_file = csv.reader(file)
        for data in csv_file:
            print(data)


def read_csv_lastline(path):
    with open(path, "r+") as file:
        mLines = file.readlines()
        targetLine = mLines[-1]
        a = targetLine.split(',')
        return a
