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
    "Returns the first item in coll. For dictlikes, returns the first k, v tuple."
    if hasattr(coll, 'iteritems'):
        return coll.iteritems().next()

    elif hasattr(coll, 'items'):
        return coll.items()[0]

    elif hasattr(coll, 'next'):
        return coll.next()

    elif hasattr(coll, '__getslice__'):
        return coll[0]

    elif hasattr(coll, '__iter__'):
        return iter(coll).next()

    return coll[0]
