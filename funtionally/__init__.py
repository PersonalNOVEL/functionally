#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools

from itertools import imap, ifilter, islice, izip, repeat


def identity(x):
    "Returns x unchanged."
    return x


def const(x, *args, **kw):
    "Returns x unchanged, ignoring all other arguments."
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


def delegate(func, *args):
    """ Takes a function and its regular positional arguments but the first,
        and returns a one-argument function (that can still take keyword args).

        >>> delegate(isinstance, unicode)(u"Hello!")
        True
        >>> delegate(getattr, 'real')(1+1j)
        1.0
        >>> delegate(int)("2")
        2
    """
    return lambda x, **kw: func(x, *args, **kw)


def complement(func):
    """ Returns a function that takes the same arguments as `func`, but returns
        the negated boolean truth value of `func`'s result.

        >>> even = lambda x: x % 2 == 0
        >>> even(2), even(3)
        (True, False)
        >>> odd = complement(even)
        >>> odd(2), odd(3)
        (False, True)
    """
    def _complement(*args, **kw):
        return not func(*args, **kw)
    update_wrapper(_complement, func)
    try:
        _complement.__name__ = _complement.func_name = 'complement_'+func.__name__
    except AttributeError:
        pass
    return _complement


def compose(*funcs):
    """ Takes a list of functions and returns a function that is the composition
        of these functions. The resulting function takes any arguments and calls the
        rightmost function with these args, calls the second-but-rightmost function
        with the result, etc.

        >>> flip_number = compose(int, ''.join, reversed, str)
        >>> flip_number(7331)
        1337
        >>> flip_number.__name__
        'composed_int_join_reversed_str'
        >>> str2 = compose(str)
        >>> str2(u'Hello World')
        'Hello World'
    """
    funcs = list(funcs)
    assert funcs, 'compose needs at least one argument'

    def composed(*a, **k):
        result = funcs[-1](*a, **k)
        for func in reversed(funcs[:-1]):
            result = func(result)
        return result

    composed.__name__ = 'composed_%s' % \
        '_'.join(imap(lambda f: getattr(f, '__name__', '???'), funcs))
    return composed


def thrush(x, *funcs):
    """ Pipes x through funcs in the given order. That is, applies the first
        func to x, applies the second func to the result, etc.

        >>> thrush(1, lambda x: x+2, lambda x: x/2.0)
        1.5
        >>> thrush(1, lambda x: x/2.0, lambda x: x+2)
        2.5
        >>> thrush('no-op')
        'no-op'
    """
    return reduce(lambda res, f: f(res), funcs, x)


def update_wrapper(wrapper, wrapped, assigned=functools.WRAPPER_ASSIGNMENTS,
                   updated=functools.WRAPPER_UPDATES):
    """ Update a wrapper function to look like the wrapped function.

        Modified version to support partial and other non-__dict__ objects.
        See functools.update_wrapper for full documentation.
    """
    for attr in assigned:
        try:
            setattr(wrapper, attr, getattr(wrapped, attr))
        except AttributeError:
            pass
    for attr in updated:
        try:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        except AttributeError:
            pass
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


def maybe(func):
    """ Turns a unary function into a nil-safe function. That is, if the wrapped
        function receives None as its argument, it will return None immediately.

    >>> float(3)
    3.0
    >>> float(None)
    Traceback (most recent call last):
    ...
    TypeError: float() argument must be a string or a number
    >>> mfloat = maybe(float)
    >>> mfloat(3)
    3.0
    >>> mfloat(None)
    """
    def _maybe(x, *args, **kw):
        if x is None:
            return None
        return func(x, *args, **kw)
    return update_wrapper(_maybe, func)


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


def somefx(f, coll):
    """ Returns the first f(x) for x in coll where f(x) is logical true.
        If no such element is found, returns None.

    >>> import string
    >>> somefx(string.strip, ['', '     ', '     Hello ', ' '])
    'Hello'
    >>> somefx(string.strip, ['', '     ', ' '])
    """
    for elem in coll:
        res = f(elem)
        if res:
            return res
    return None


def keep(f, coll):
    """ Returns an iterator containing the items x of coll for which pred(x)
        does not return None. This is in contrast to filter which also
        throws away falsy values.

    >>> list(keep(lambda x: getattr(x, 'imag', None),
    ...           [1+1j, 1+0j, 42, 0.0, None, 'foo']))
    [(1+1j), (1+0j), 42, 0.0]
    """
    return (x for x in sequify(coll) if f(x) is not None)


def filter_attr(attr, coll):
    """ Acts like ifilter(attrgetter(attr), coll).

    >>> list(filter_attr('imag', [1+0j, 1+1j, 42+0j]))
    [(1+1j)]
    """
    return (elem for elem in coll if getattr(elem, attr))


def sequify(coll):
    """ Returns an iterator on coll, just like iter(coll). dict-like objects
        will be treated as sequences of (key, value) pairs, however.

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
        return iter(coll.items())
    return iter(coll)


def first(coll):
    """ Returns the first item in coll. For dict-like objects, returns the
        first (k, v) tuple. If coll is empty, returns None.
    """

    if hasattr(coll, '__getslice__'):
        try:
            return coll[0]
        except IndexError:
            return None

    itr = sequify(coll)

    try:
        return itr.next()
    except StopIteration:
        return None

    raise NotImplementedError("Can't get first item from type %r" % type(coll).__name_)


def second(coll):
    "Like first, but returns the second item in coll."

    if hasattr(coll, '__getslice__'):
        try:
            return coll[1]
        except IndexError:
            return None

    itr = sequify(coll)

    try:
        itr.next()
        return itr.next()
    except StopIteration:
        return None

    raise NotImplementedError("Can't get second item from type %r" % type(coll).__name_)


def last(coll):
    "Like first, but returns the last item in coll."

    if hasattr(coll, '__getslice__'):
        try:
            return coll[-1]
        except IndexError:
            return None

    itr = sequify(coll)

    last = None
    try:
        while True:
            last = itr.next()
    except StopIteration:
        return last

    raise NotImplementedError("Can't get last item from type %r" % type(coll).__name_)


def rest(coll):
    """ Returns all items in coll but the first. For dict-like objects, returns
        (k, v) tuples except for the 'first' one. If coll is empty, returns None
    """

    if hasattr(coll, '__getslice__'):
        rst = coll[1:]
        if len(rst) == 0:
            return None
        return iter(rst)

    itr = sequify(coll)

    try:
        itr.next()
        return itr
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
    yield x
    for elem in rst:
        yield elem


def butlast(coll):
    """ Returns an iterable containing all elements in coll except for the last.

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


def split_at(n, coll):
    """ Returns a tuple of (coll[:n], coll[n:]).

    >>> split_at(1, ['Hallo', 'Welt'])
    (['Hallo'], ['Welt'])
    """
    return (coll[:n], coll[n:])


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
    """ Returns an iterator of elemens in seq, partitioned into tuples
        of n elements. If len(seq) is not a multiple of n, the last tuple
        will contain less than n elements.

    >>> list(partition([1, 2, 3, 4, 5, 6], 2))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(partition([1, 2, 3, 4, 5, 6], 3))
    [(1, 2, 3), (4, 5, 6)]
    >>> list(partition([1, 2, 3, 4, 5], 2))
    [(1, 2), (3, 4), (5,)]
    >>> list(partition([], 2))
    []
    """
    seq = iter(seq)
    while True:
        part = tuple(islice(seq, n))
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
    """ Returns an iterator of elemens in seq, partitioned so that elements
        will appear in the first column, then the second column, etc. If len(seq)
        is not a multiple of n, the last row will contain less than n columns.

        >>> res = vertical_partition(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E',
        ...                           'E', 'F', 'F', 'G', 'G', 'H', 'H', 'I'], 4)
        >>> assert res == [['A', 'C', 'E', 'G'],
        ...                ['A', 'D', 'F', 'H'],
        ...                ['B', 'D', 'F', 'H'],
        ...                ['B', 'E', 'G', 'I'],
        ...                ['C']]
    """
    p = [list(part) for part in partition(seq, n)]
    seq = iter(seq)

    for ni in xrange(n):
        for ri in xrange(len(p)):
            if len(p[ri]) > ni:
                p[ri][ni] = seq.next()
    return p


def map_all(funcs, iterable):
    """ Takes an iterable of functions (callables) and an iterable.
        Applies every function in the specified order to each element
        in the iterable and returns the result as a list.

        >>> map_all([str, reversed, ''.join, int],
        ...         [7331, 58008])
        [1337, 80085]
    """
    for func in funcs:
        iterable = imap(func, iterable)
    return list(iterable)


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


def filter_values(pred, d):
    """ Returns a new dict with only those k, v pairs where pred(v) holds.

    >>> D = {'a': 1, 'b': 2, 'c': 3}
    >>> odd = lambda x: x % 2 != 0
    >>> filter_values(odd, D)
    {'a': 1, 'c': 3}
    """
    return dict((k, v) for k, v in d.iteritems() if pred(v))


def interleave(*colls):
    """ Returns an iterable yielding the first element of all colls in turn,
        then the second, etc. until the shortest coll is exhausted.

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
    """ Returns an iterable containing all elements in coll separated by sep.

    >>> list(interpose('and', [1, 2, 3]))
    [1, 'and', 2, 'and', 3]
    >>> list(interpose(42, []))
    []
    """
    return butlast(interleave(coll, repeat(sep)))


def strcat(*xs):
    """ Concatenates all strings passed as positional arguments.

    >>> strcat('Brave ', 'New ', 'World', '!')
    'Brave New World!'
    >>> strcat('char', u'unicode')
    u'charunicode'
    >>> strcat()
    ''
    >>> strcat('Value: ', 1)
    Traceback (most recent call last):
    ...
    TypeError: strcat arguments must be strings, not int
    """
    res = ""
    for x in xs:
        if not isinstance(x, basestring):
            raise TypeError("strcat arguments must be strings, "
                            "not %s" % type(x).__name__)
        res += x
    return res


def coalesce(*args):
    """ Returns the first argument that is not None.

    >>> coalesce(None, None, 1, 2, None)
    1
    >>> coalesce(3, 2, 1)
    3
    >>> coalesce(None, None, None)
    >>> coalesce()
    """
    return some(lambda x: x is not None, args)


def treeseq(is_branch, children_fn, root):
    """ Turns a tree structure into a sequence. `is_branch` should be a unary
        predicate function that determines whether a node can have children,
        and `children_fn` should return these children when applied to the node.

    >>> from functools import partial
    >>> listseq = partial(treeseq, delegate(isinstance, list), identity)
    >>> list(listseq([1, [2, 3, [4]], 5]))
    [[1, [2, 3, [4]], 5], 1, [2, 3, [4]], 2, 3, [4], 4, 5]
    """
    def walk(node):
        yield node
        if is_branch(node):
            for child in children_fn(node):
                for x in walk(child):
                    yield x
    for x in walk(root):
        yield x


def treeleaves(is_branch, children_fn, root):
    """ Like treeseq, but only yields the tree's leaf nodes.

    >>> from functools import partial
    >>> listleaves = partial(treeleaves, delegate(isinstance, list), identity)
    >>> list(listleaves([1, [2, 3, [4]], 5]))
    [1, 2, 3, 4, 5]
    >>> dictleaves = partial(treeleaves, delegate(isinstance, dict), \
                             lambda d: d.itervalues())
    >>> sorted(dictleaves({'a': 1, 'b': {'c': {'d': 2, 'e': 3}}, 42: 4, 'f': 5}))
    [1, 2, 3, 4, 5]
    """
    nodes = treeseq(is_branch, children_fn, root)
    return ifilter(complement(is_branch), nodes)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '--stop', '--with-doctest', '-vvs'])
