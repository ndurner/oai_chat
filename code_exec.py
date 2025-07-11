from RestrictedPython import compile_restricted
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Guards import (
    safe_globals,
    safe_builtins,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
    full_write_guard,
)
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Utilities import utility_builtins
from RestrictedPython.Limits import limited_range, limited_list, limited_tuple
from io import StringIO
import types
import sys
import os
from pathlib import Path


def unrestricted_python_enabled() -> bool:
    """Check if UnrestrictedPython mode is enabled."""
    val = os.environ.get("CODE_EXEC_UNRESTRICTED_PYTHON")
    if val is not None:
        return val == "1"
    env_path = Path(".env")
    if env_path.is_file():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, sep, v = line.partition("=")
                if k == "CODE_EXEC_UNRESTRICTED_PYTHON" and sep:
                    return v.strip().strip('"').strip("'") == "1"
    return False

def eval_script(script):
    """Evaluate Python script according to mode."""
    if unrestricted_python_enabled():
        return eval_unrestricted_script(script)
    return eval_restricted_script(script)

def eval_restricted_script(script):

    # Set up print collector and output handling
    all_prints = StringIO()
    
    class CustomPrintCollector:
        """Collect printed text, accumulating in shared StringIO"""
        
        def __init__(self, _getattr_=None):
            self.txt = []
            self._getattr_ = _getattr_
        
        def write(self, text):
            all_prints.write(text)
            self.txt.append(text)
            
        def __call__(self):
            result = ''.join(self.txt)
            return result
            
        def _call_print(self, *objects, **kwargs):
            if kwargs.get('file', None) is None:
                kwargs['file'] = self
            else:
                self._getattr_(kwargs['file'], 'write')
            
            print(*objects, **kwargs)

    # Create the restricted builtins dictionary
    restricted_builtins = dict(safe_builtins)
    restricted_builtins.update(utility_builtins)  # Add safe __import__
    restricted_builtins.update({
        # Print handling
        '_print_': CustomPrintCollector,
        '_getattr_': safer_getattr,
        '_getitem_': default_guarded_getitem,
        '_getiter_': default_guarded_getiter,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        '_unpack_sequence_': guarded_unpack_sequence,
        '_inplacevar_': protected_inplacevar,
        '_apply_': _apply,
        '_write_': _write_guard,
        
        # Define allowed imports
        '__allowed_modules__': ['math', 'datetime', 'time'],
        '__import__': safe_import,
        
        # Basic functions
        'len': len,
        'range': limited_range,
        'enumerate': enumerate,
        'zip': zip,
        
        # Math operations
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'pow': pow,
        
        # Type conversions
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': limited_list,
        'tuple': limited_tuple,
        'set': set,
        'dict': dict,
        'bytes': bytes,
        'bytearray': bytearray,
        
        # Sequence operations
        'all': all,
        'any': any,
        'sorted': sorted,
        'reversed': reversed,
        
        # String operations
        'chr': chr,
        'ord': ord,
        
        # Other safe operations
        'isinstance': isinstance,
        'issubclass': issubclass,
        'hasattr': hasattr,
        'callable': callable,
        'format': format,
    })

    # Create the restricted globals dictionary
    restricted_globals = dict(safe_globals)
    restricted_globals['__builtins__'] = restricted_builtins

    try:
        byte_code = compile_restricted(script, filename='<inline>', mode='exec')
        exec(byte_code, restricted_globals)
        
        return {
            'prints': all_prints.getvalue(),
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def eval_unrestricted_script(script):
    """Execute Python script without sandboxing."""
    output = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = output
        exec(script, {})
        return {
            'prints': output.getvalue(),
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }
    finally:
        sys.stdout = old_stdout

def _write_guard(obj):
    """Stricter write guard that wraps objects using full_write_guard."""
    if isinstance(obj, types.ModuleType):
        raise ValueError("Modules are not writable in the sandbox.")
    return full_write_guard(obj)

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed = ['math', 'datetime']
    if name not in allowed:
        raise ImportError(f"Import of module '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)

""" 
Borrowed implementation of _inplacevar_ from the Zope Foundations's AccessControl module
https://github.com/zopefoundation/AccessControl/blob/f9ae58816f0712eb6ea97459b4ccafbf4662d9db/src/AccessControl/ZopeGuards.py#L530
"""

valid_inplace_types = (list, set)


inplace_slots = {
    '+=': '__iadd__',
    '-=': '__isub__',
    '*=': '__imul__',
    '/=': (1 / 2 == 0) and '__idiv__' or '__itruediv__',
    '//=': '__ifloordiv__',
    '%=': '__imod__',
    '**=': '__ipow__',
    '<<=': '__ilshift__',
    '>>=': '__irshift__',
    '&=': '__iand__',
    '^=': '__ixor__',
    '|=': '__ior__',
}


def __iadd__(x, y):
    x += y
    return x


def __isub__(x, y):
    x -= y
    return x


def __imul__(x, y):
    x *= y
    return x


def __idiv__(x, y):
    x /= y
    return x


def __ifloordiv__(x, y):
    x //= y
    return x


def __imod__(x, y):
    x %= y
    return x


def __ipow__(x, y):
    x **= y
    return x


def __ilshift__(x, y):
    x <<= y
    return x


def __irshift__(x, y):
    x >>= y
    return x


def __iand__(x, y):
    x &= y
    return x


def __ixor__(x, y):
    x ^= y
    return x


def __ior__(x, y):
    x |= y
    return x


inplace_ops = {
    '+=': __iadd__,
    '-=': __isub__,
    '*=': __imul__,
    '/=': __idiv__,
    '//=': __ifloordiv__,
    '%=': __imod__,
    '**=': __ipow__,
    '<<=': __ilshift__,
    '>>=': __irshift__,
    '&=': __iand__,
    '^=': __ixor__,
    '|=': __ior__,
}


def protected_inplacevar(op, var, expr):
    """Do an inplace operation

    If the var has an inplace slot, then disallow the operation
    unless the var an instance of ``valid_inplace_types``.
    """
    if hasattr(var, inplace_slots[op]) and \
    not isinstance(var, valid_inplace_types):
        try:
            cls = var.__class__
        except AttributeError:
            cls = type(var)
        raise TypeError(
            "Augmented assignment to %s objects is not allowed"
            " in untrusted code" % cls.__name__)
    return inplace_ops[op](var, expr)

def _apply(f, *a, **kw):
    return f(*a, **kw)
