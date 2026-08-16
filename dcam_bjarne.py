"""
dcam_fix.py
Runtime fix for the heap corruption in Hamamatsu's dcamapi4.py. No vendor files are edited.

THE BUG.  Both DCAMDEV_STRING.alloctext (dcamapi4.py:550) and DCAMPROP_VALUETEXT.alloctext
(dcamapi4.py:1111) do:

    def alloctext(self, maxlen):
        textbuf = create_string_buffer(maxlen)
        self.text = addressof(textbuf)      # plain int -> ctypes keeps NO reference
        self.textbytes = sizeof(textbuf)

`textbuf` is a local and `addressof()` returns an integer, so ctypes stores the number and
keeps no keepalive reference (struct._objects is None). The buffer is freed the moment
alloctext returns, and the DCAM DLL then writes 256 bytes of device string into freed heap.
getDeviceString() loops over every DCAM_IDSTR, so that is ~10 stomps per call, and
ImageAcquisition.run() calls it at the top -- i.e. every time you press Run.

Python 3.7 usually survived it because the freed block had not been recycled yet. 3.11+
reuses blocks aggressively, so the *next* allocation walks a corrupted free list and the
process dies with an access violation inside create_string_buffer. That is exactly the
traceback we saw: it faults on the second Run, not the first.

THE FIX.  Anchor the buffer on the struct instance so it lives exactly as long as the thing
pointing into it. ctypes Structures carry a normal __dict__, so the extra attribute is fine.
Patching the class covers both dcam.py call sites (dev_getstring, prop_getvaluetext) at
once, because `from dcamapi4 import *` binds the same class objects.

USE.  Import this before Dcamapi.init() and before any thread starts:

    import dcam_fix            # must come before the first camera call
    dcam_fix.apply()

@author: patch for Bjarne Schuemann's setup
"""

from ctypes import addressof, create_string_buffer, sizeof

_applied = False


def _alloctext(self, maxlen):
    """alloctext with the buffer anchored on the instance (keeps it alive)."""
    self._textbuf = create_string_buffer(maxlen)
    self.text = addressof(self._textbuf)
    self.textbytes = sizeof(self._textbuf)


def apply(verbose=True):
    """Patch every DCAM struct that allocates a text buffer. Idempotent."""
    global _applied
    if _applied:
        return True
    try:
        import dcamapi4
    except ImportError as e:
        print("[dcam_bjarne] could not import dcamapi4 (%s) - camera stack not on sys.path?" % e)
        return False

    patched = []
    for name in ("DCAMDEV_STRING", "DCAMPROP_VALUETEXT"):
        cls = getattr(dcamapi4, name, None)
        if cls is None:
            print("[dcam_bjarne] %s not found in dcamapi4 - skipped." % name)
            continue
        if not hasattr(cls, "alloctext"):
            print("[dcam_bjarne] %s has no alloctext - vendor file changed, re-check." % name)
            continue
        cls.alloctext = _alloctext
        patched.append(name)

    # Anything else in the module that allocates a text buffer the same way.
    for name in dir(dcamapi4):
        cls = getattr(dcamapi4, name)
        if (
            isinstance(cls, type)
            and name not in patched
            and getattr(cls, "alloctext", None) not in (None, _alloctext)
        ):
            cls.alloctext = _alloctext
            patched.append(name)

    _applied = bool(patched)
    if verbose:
        print("[dcam_bjarne] alloctext patched on: %s" % (", ".join(patched) or "nothing"))
    return _applied


def selftest():
    """Confirm the unpatched pattern really does drop the buffer, on this interpreter."""
    from ctypes import Structure, c_char_p, c_int32

    class _S(Structure):
        _pack_ = 8
        _fields_ = [("size", c_int32), ("iString", c_int32),
                    ("text", c_char_p), ("textbytes", c_int32)]

    s = _S()
    b = create_string_buffer(64)
    s.text = addressof(b)
    print("[dcam_bjarne] keepalive after raw addressof assignment: %r  (None == the bug)"
          % (s._objects,))
    _alloctext(s, 64)
    print("[dcam_bjarne] buffer anchored on instance: %r" % (hasattr(s, "_textbuf"),))


if __name__ == "__main__":
    selftest()
    apply()
