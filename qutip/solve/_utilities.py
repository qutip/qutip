import os


def cython_build_cleanup(tdname, build_dir=None):
    if build_dir is None:
        build_dir = os.path.join(os.path.expanduser('~'), '.pyxbld')
    # Remove tdname.pyx
    pyx_file = tdname + ".pyx"
    try:
        os.remove(pyx_file)
    except OSError:
        pass
    # Remove temp build files
    for dirpath, _, files in os.walk(build_dir):
        for f in files:
            if f.startswith(tdname):
                try:
                    os.remove(os.path.join(dirpath, f))
                except OSError:
                    pass
