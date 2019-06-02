import numpy as np

from functools import reduce, partial

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

arrayt = compose(np.array, tuple)
arrayl = compose(np.array, list)

def mapt(f, l):
    return compose(arrayt, partial(map, f))(l)

def mapl(f, l):
    return compose(arrayl, partial(map, f))(l)

def countif(f, l):
    return compose(len, arrayt, partial(filter, f))(l)

def append(acc, l):
    return np.append(acc, l, axis=0)

def add(x, y):
    return x+y

def identity(x):
	return x

def flatten(x):
    return reduce(append, x)

def add_key(dic, key, val):
    dic[key] = val
    return dic
