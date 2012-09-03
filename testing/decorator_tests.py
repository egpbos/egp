import functools

class TestDecorator(object):
    """
    
    """
    def __init__(self, fct, *args, **kwargs):
        self.fct = fct
        self.args = args
        self.kwargs = kwargs
        print args
        print kwargs
    
    def __call__(self, *args, **kwargs):
        print args
        print kwargs
    
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


@TestDecorator()
def test_function(one, two, three):
    print "blabla"

@TestDecorator("piep", 1, 4, 16)
def test_function(one, two, three):
    print "blabla"

@TestDecorator
def test_function(one, two, three):
    print "blabla"



class TraceCalls(object):
    """ Use as a decorator on functions that should be traced. Several
        functions can be decorated - they will all be indented according
        to their call depth.
    """
    def __init__(self, stream=sys.stdout, indent_step=2, show_ret=False):
        self.stream = stream
        self.indent_step = indent_step
        self.show_ret = show_ret

        # This is a class attribute since we want to share the indentation
        # level between different traced functions, in case they call
        # each other.
        TraceCalls.cur_indent = 0

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            indent = ' ' * TraceCalls.cur_indent
            argstr = ', '.join(
                [repr(a) for a in args] +
                ["%s=%s" % (a, repr(b)) for a, b in kwargs.items()])
            self.stream.write('%s%s(%s)\n' % (indent, fn.__name__, argstr))

            TraceCalls.cur_indent += self.indent_step
            ret = fn(*args, **kwargs)
            TraceCalls.cur_indent -= self.indent_step

            if self.show_ret:
                self.stream.write('%s--> %s\n' % (indent, ret))
            return ret
        return wrapper
        

def set_k_abs_grid(caching_active = False):
    @cache(active=caching_active, cache_key = "grid_%s_box_%s")
    def fct(gridsize, boxlen):
        pass # de code
    global k_abs_grid = fct

set_k_abs_grid() 

class Cacheable(object):
    """
    Caching is disabled by default, but can be enabled after importing the
    decorated function by calling the cache_on() method on the decorated
    function.
    """
    def __init__(self, cache_key = None):
        self.cache_key = cache_key
    
    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            

def make(*args):
    print locals()
    def wrapper(f):
        print ">", f, args
        return TestDecorator(f, *args)
    return wrapper

@make(1)
def test_function1(one, two, three):
 print "blabla"