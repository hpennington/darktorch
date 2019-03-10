#!/usr/bin/env python3


def parse_data(filepath):
    fd = open(filepath, 'r')
    lines = fd.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    fd.close()

    config = {}

    for line in lines:
        key, val = line.split("=")
        config[key.rstrip()] = val.lstrip()
    return config


def parse_categories(filepath):
    fd = open(filepath, 'r')
    lines = fd.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    fd.close()

    categories = []

    for line in lines:
        categories.append(line.strip())
    return categories


def parse_configuration(filepath):
    cfg_block = {}
    blocks = []
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip()

        if line.startswith('['):
            if len(cfg_block) != 0:
                blocks.append(cfg_block)
                cfg_block = {}
            line = line.replace('[', '').replace(']', '')
            cfg_block['type'] = line

        elif line.startswith('#'):
            continue

        elif '=' in line:
            key, val = line.split('=')
            key, val = key.strip(), val.strip()
            cfg_block[key] = val

    blocks.append(cfg_block)

    return blocks
