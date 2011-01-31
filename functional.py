#!/usr/bin/env python
# -*- coding: utf-8 -*-

from operator import attrgetter
from itertools import chain, imap, islice, izip, repeat

def identity(x):
    "Returns x unchanged."
    return x

def constantly(result):
    """ Returns a function that will take any arguments, ignore them and always
        return `result`.

        >>> deaf_guy = constantly('what?')
        >>> deaf_guy("Penguin")
        'what?'
        >>> deaf_guy("Penguin!", "You know.", those="guy's in suits!")
        'what?'

    """
    def constantly_wrapper(*args, **kw):
        return result
    return constantly_wrapper

def compose(*funcs):
    """ Takes a list of functions and returns a function that is the composition
        of these functions. The resulting function takes any arguments and calls the
        rightmost function with these args, calls the second-but-rightmost function
        with the result, etc.

        >>> flip_number = compose(str, reversed, ''.join, int)
        >>> flip_number(7331)
        1337
        >>> flip_number.__name__
        'composed_str_reversed_join_int'
        >>> str2 = compose(str)
        >>> str2(u'Hello World')
        'Hello World'

    """
    funcs = list(funcs)
    assert funcs, 'compose needs at least one argument'

    def composed(*a, **k):
        result = funcs[0](*a, **k)
        for func in funcs[1:]:
            result = func(result)
        return result

    composed.__name__ = 'composed_%s' % (
        '_'.join(map(attrgetter('__name__'), funcs)))
    return composed


def some(pred, coll):
    """ Returns the first element x in coll where pred(x) is logical true.
        If no such element is found, returns None.

        >>> fish_are_blue = lambda x: "blue" in x.lower() and "fish" in x.lower()
        >>> some(fish_are_blue,
        ...      ["Red fish", "Green fish", "Blue fish", "Blue and yellow fish"])
        'Blue fish'
        >>> some(fish_are_blue,
        ...      ["Red dog", "Green dog", "Blue dog", "Blue and yellow fish"])
        'Blue and yellow fish'
        >>> some(fish_are_blue,
        ...      ["Red dog", "Green dog", "Blue dog"])

    """
    for elem in coll:
        if pred(elem):
            return elem
    return None

def sequify(coll):
    """ Returns an iterator on coll, just like iter(coll). dict-like object will
        be treated as sequences of (key, value) pairs, however.

    >>> list(sequify([]))
    []
    >>> list(sequify({'a': 1, 'b': 2}))
    [('a', 1), ('b', 2)]
    >>> list(sequify('foo'))
    ['f', 'o', 'o']
    >>> it = sequify([1, 2])
    >>> it.next()
    1
    """
    if hasattr(coll, 'iteritems'):
        return coll.iteritems()
    elif hasattr(coll, 'items'):
        return coll.items()
    return iter(coll)

def first(coll):
    """ Returns the first item in coll. For dict-like objects, returns the
        first (k, v) tuple. If coll is empty, returns None.
    """
    coll = sequify(coll)

    if hasattr(coll, '__getslice__'):
        try:
            return coll[0]
        except IndexError:
            return None

    elif hasattr(coll, '__iter__'):
        try:
            return iter(coll).next()
        except StopIteration:
            return None

    raise NotImplementedError("Can't get first item from type %r" % type(coll).__name_)

def second(coll):
    "Like first, but returns the second item in coll."
    coll = sequify(coll)

    if hasattr(coll, '__getslice__'):
        try:
            return coll[1]
        except IndexError:
            return None

    elif hasattr(coll, '__iter__'):
        try:
            itr = iter(coll)
            itr.next()
            return itr.next()
        except StopIteration:
            return None

    raise NotImplementedError("Can't get first item from type %r" % type(coll).__name_)

def rest(coll):
    """ Returns all items in coll but the first. For dict-like objects, returns
        (k, v) tuples except for the 'first' one. If coll is empty, returns None
    """
    coll = sequify(coll)

    if hasattr(coll, '__getslice__'):
        rst = coll[1:]
        if len(rst) == 0:
            return None
        return rst

    elif hasattr(coll, 'next'):
        try:
            coll.next()
            return coll
        except StopIteration:
            return None

    elif hasattr(coll, '__iter__'):
        try:
            return rest(iter(coll))
        except StopIteration:
            return None

    raise NotImplementedError("Can't get rest from type %r" % type(coll).__name_)

def cons(x, rst):
    """ Returns a new iterable where x is the first element and rst is the rest.

    >>> list(cons(1, [2, 3]))
    [1, 2, 3]
    >>> list(cons(1, iter([])))
    [1]
    >>> list(cons('a', 'bc'))
    ['a', 'b', 'c']
    """
    return chain(iter([x]), rst)

def butlast(coll):
    """
    >>> list(butlast([1, 2, 3]))
    [1, 2]
    >>> list(butlast({'foo': 1, 'bar': 2}))
    [('foo', 1)]
    >>> list(butlast('foo'))
    ['f', 'o']
    """
    seq = sequify(coll)
    prev = seq.next()
    while True:
        try:
            curr = seq.next()
        except StopIteration:
            return
        yield prev
        prev = curr

def take(n, coll):
    """ Returns the first n elements from coll.

    >>> list(take(2, [1, 2, 3, 4]))
    [1, 2]
    >>> list(take(0, [1, 2, 3, 4]))
    []
    """
    for elem, _ in izip(sequify(coll), xrange(n)):
        yield elem

def drop(n, coll):
    """ Returns all elements in coll but the first n.

    >>> list(drop(2, [1, 2, 3, 4]))
    [3, 4]
    >>> list(drop(0, [1, 2, 3, 4]))
    [1, 2, 3, 4]
    """
    coll = sequify(coll)
    for _ in xrange(n):
        coll.next()
    for elem in coll:
        yield elem


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

         partition([1, 2, 3, 4, ...], 2) => [(1, 2), (3, 4), ...]
    """
    seq = iter(seq)
    while True:
        part = list(islice(seq, n))
        if not part:
            return
        yield part

def partition_by(f, coll):
    u""" Splits coll into partitions, beginning a new partition each time f(x)
         returns a new value than for the previous element.

    >>> list(partition_by(lambda x: x == 3, [1, 2, 3, 4, 5]))
    [(1, 2), (3,), (4, 5)]
    >>> list(partition_by(lambda x: bool(x%2), [1, 3, 5, 2, 4, 6, 7]))
    [(1, 3, 5), (2, 4, 6), (7,)]
    >>> list(partition_by(identity, []))
    []
    """
    # HACK: OMG, schlecht implementiert. Das geht alles viel einfacher.
    fst = first(coll)
    rst = rest(coll)
    fprev = f(fst)
    part = [fst]

    if rst is None:
        return

    for elem in rst:
        fnow = f(elem)
        if fnow != fprev:
            yield tuple(part)
            part = []
        part.append(elem)
        fprev = fnow

    if part:
        yield tuple(part)


def vertical_partition(seq, n):
    """
        >>> res = vertical_partition(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E',
        ...                           'E', 'F', 'F', 'G', 'G', 'H', 'H', 'I'], 4)
        >>> assert res == [['A', 'C', 'E', 'G'],
        ...                ['A', 'D', 'F', 'H'],
        ...                ['B', 'D', 'F', 'H'],
        ...                ['B', 'E', 'G', 'I'],
        ...                ['C']]

    """
    p = list(partition(seq, n))

    def mk_seq(lst):
        for i in lst:
            yield i

    seq = mk_seq(seq)

    for ni in xrange(n):
        for ri in xrange(len(p)):
            if len(p[ri]) > ni:
                p[ri][ni] = seq.next()
    return p

def map_all(funcs, iterable):
    """Takes an iterable of functions (callables) and an iterable.
       Applies every function in the specified order to each element
       in the iterable and returns the result as a list.

        >>> map_all([str, reversed, ''.join, int],
        ...         [7331, 58008])
        [1337, 80085]

        >>> from lib.utils.dimens import unit, point
        >>> inch = unit("inch")
        >>> map_all((float, inch, point, float), [2.54, 1])
        [182.88, 72.0]
    """
    for func in funcs:
        iterable = map(func, iterable)
    return iterable

def map_keys(func, d):
    """ Returns a new dict with func applied to keys from d, while values
        remain unchanged.
    >>> D = {'a': 1, 'b': 2}
    >>> map_keys(lambda k: k.upper(), D)
    {'A': 1, 'B': 2}
    >>> assert map_keys(identity, D) == D
    >>> map_keys(identity, {})
    {}

    """
    return dict((func(k), v) for k, v in d.iteritems())

def interleave(*colls):
    """ Returns an iterator yielding the first element of all colls in turn,
        then the second , etc. until the shortest coll runs out.

    >>> list(interleave([1, 2, 3], [4, 5]))
    [1, 4, 2, 5]
    >>> list(interleave([1, 4], [2, 5], [3, 6]))
    [1, 2, 3, 4, 5, 6]
    >>> list(interleave([1, 2, 3], []))
    []
    >>> list(interleave([], [], []))
    []
    """
    colls = map(sequify, colls)
    while True:
        nexts = [c.next() for c in colls]
        for elem in nexts:
            yield elem


def interpose(sep, coll):
    """ Returns all elements in coll separated by sep

    >>> list(interpose('and', [1, 2, 3]))
    [1, 'and', 2, 'and', 3]
    >>> list(interpose(42, []))
    []
    """
    return butlast(interleave(coll, repeat(sep)))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '--stop', '--with-doctest', '-vvs'])
