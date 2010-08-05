#!/usr/bin/env python
# -*- coding: utf-8 -*-

def identity(x):
    "Returns x unchanged."
    return x

def constantly(result):
    """ Returns a function that takes any arguments, ignores them and always
        returns `result`.
    """
    def constantly_wrapper(*args, **kw):
        return result
    return constantly_wrapper

def some(pred, coll):
    "Returns the first element x in coll where pred(x) is logical true, otherwise None."

    for elem in coll:
        if pred(elem):
            return elem
    return None

def first(coll):
    """Returns the first item in coll. For dictlikes, returns the first k, v tuple.
       Return None if coll is empty.
    """
    if hasattr(coll, 'iteritems'):
        try:
            return coll.iteritems().next()
        except StopIteration:
            return None

    elif hasattr(coll, 'items'):
        try:
            return coll.items()[0]
        except IndexError:
            return None

    elif hasattr(coll, 'next'):
        try:
            return coll.next()
        except StopIteration:
            return None

    elif hasattr(coll, '__getslice__'):
        try:
            return coll[0]
        except IndexError:
            return None

    elif hasattr(coll, '__iter__'):
        try:
            return iter(coll).next()
        except StopIteration:
            return None

    raise NotImplementedError("Type %r not supported by first" % type(coll))
