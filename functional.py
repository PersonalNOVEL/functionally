#!/usr/bin/env python
# -*- coding: utf-8 -*-

def identity(x):
    "Returns x unchanged."
    return x

def some(pred, coll):
    "Returns the first element x in coll where pred(x) is logical true, otherwise None."

    for elem in coll:
        if pred(elem):
            return elem
    return None

def first(coll):
    if hasattr(coll, 'iteritems'):
        return coll.iteritems().next()

    elif hasattr(coll, 'items'):
        return coll.items[0]

    return coll[0]
