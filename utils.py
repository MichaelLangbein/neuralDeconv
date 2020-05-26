

def tabling(f):
    tabling.table = []
    def wrapped(*args, **kwargs):
        for (targs, tkwargs), tresult in tabling.table:
            if args == targs and kwargs == tkwargs:
                return tresult
        result = f(*args, **kwargs)
        tabling.table.append(((args, kwargs), result))
        return result
    wrapped.__name__ = 'w' + f.__name__
    return wrapped


def logging(f):
    logging.depth = 0
    def wrapped(*args, **kwargs):
        print(f"{'|---' * logging.depth}{f.__name__}{args}")
        logging.depth += 1
        result = f(*args, **kwargs)
        logging.depth -= 1
        print(f"{'|---' * logging.depth}{f.__name__}{args} => {result}")
        return result
    return wrapped


