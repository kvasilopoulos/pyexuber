"""Windows CI/dev-only DLL search path plumbing.

Since Python 3.8, extension-module DLL dependency resolution on Windows no
longer consults PATH at all (bpo-36085) -- os.add_dll_directory() is
required instead. Needed here because _core.pyd dynamically links against
vcpkg's openblas/lapack/armadillo DLLs, which aren't next to the
interpreter or bundled with the extension (that bundling, e.g. via
delvewheel, is a real-wheel packaging concern for later, not done here).
"""

import os

if os.name == "nt":
    dll_dir = os.environ.get("EXUBER_DLL_DIR")
    if dll_dir:
        os.add_dll_directory(dll_dir)
