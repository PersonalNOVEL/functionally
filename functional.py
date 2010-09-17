#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import imap, islice

def identity(x):
    "Returns x unchanged."
    return x

def constantly(result):
    """ Returns a function that will take any arguments, ignore them and always
        return `result`.
    """
    def constantly_wrapper(*args, **kw):
        return result
    return constantly_wrapper

def some(pred, coll):
    """ Returns the first element x in coll where pred(x) is logical true.
        If no such element is found, returns None.
    """
    for elem in coll:
        if pred(elem):
            return elem
    return None

def first(coll):
    """ Returns the first item in coll. For dict-like objects, returns the
        first (k, v) tuple. If coll is empty, returns None.
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

def starchain(coll_of_colls):
    """ Like itertools.chain, but takes the iterables from a containing iterable
        instead of *args. This allows coll_of_colls to be lazy.
    """
    for coll in coll_of_colls:
        for elem in coll:
            yield elem

def mapcat(func, *colls):
    """ Returns the lazy concatenation of the results from map(func, *coll).
        Thus, func should return an iterable.
    """
    return starchain(imap(func, *colls))

def partition(seq, n):
    u""" Returns an iterator of elemens in seq, partitioned into tuples
         of n elements. If len(seq) is not a multiple of n, the last tuple
         will contain less than n elements.

    >>> list(partition([1, 2, 3, 4, 5], 3))
    [(1, 2, 3), (4, 5)]
    >>> list(partition([]))
    []
    """
    seq = iter(seq)
    while True:
        part = tuple(islice(seq, n))
        if not part:
            return
        yield part
