# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:LiShuai
@Blog(个人博客地址): https://www.cnblogs.com/L-shuai/
 
@File:test01.py
@Time:2022/1/17 16:40
 
"""


def test1():
    word2idx = {}
    word2idx['a'] = 1
    print(len(word2idx))
    word2idx[99]=3
    print(len(word2idx))


if __name__ == '__main__':
    test1()