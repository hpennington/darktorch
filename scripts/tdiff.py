#!/usr/bin/env python3

import sys
import shutil
import numpy as np


def parse(fp):
    with open(fp, 'r') as f:
        x = f.read().splitlines()
        x = [np.float32(y) for y in x]
        x = np.array(x)
    return x


def parse_binary(path, dtype):
    return np.fromfile(path, dtype=dtype)

def format_cli_header(arg0, arg1):
    content = ' {} vs {} '.format(arg0.upper(), arg1.upper())
    terminal_width = shutil.get_terminal_size()[0]
    append_front = True
    while (terminal_width - len(content) > 0):
        if append_front == True:
            content = '=' + content
        else:
            content = content + '='
        append_front = not append_front
    return content


args = sys.argv

if args[1] == '-b':
    print(format_cli_header(args[3], args[4]))
    
    dtype = None

    if args[2] == 'float':
        dtype = np.float32
    elif args[2] == 'int':
        dtype = np.uint8
    else:
        print('dtype is required with binary option -b')
        sys.exit(1)
    print('BINARY mode dtype:', dtype)
    a = parse_binary(args[3], dtype)
    b = parse_binary(args[4], dtype)

else:
    print(format_cli_header(args[1], args[2]))
    print('TEXT mode')
    a = parse(args[1])
    b = parse(args[2])
    

print('a len:', len(a))
print('b len:', len(b))
print('a sum:', np.sum(a))
print('b sum:', np.sum(b))

c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 0

max_iter = len(a) if len(a) <= len(b) else len(b)
for i in range(max_iter):
    delta = a[i] - b[i]

    if a[i] == b[i]:
        c0 += 1
    if abs(delta) > 0.000001:
        c1 += 1
    if abs(delta) > 0.00001:
        c2 += 1
    if abs(delta) > 0.0001:
        c3 += 1
    if abs(delta) > 0.001:
        c4 += 1
    if abs(delta) > 0.01:
        c5 += 1
    if abs(delta) > 0.1:
        c6 += 1

print(str(c0) + ' exact matches')
print(str(c1) + ' differences greater than 0.000001')
print(str(c2) + ' differences greater than 0.00001')
print(str(c3) + ' differences greater than 0.0001')
print(str(c4) + ' differences greater than 0.001')
print(str(c5) + ' differences greater than 0.01')
print(str(c6) + ' differences greater than 0.1')
