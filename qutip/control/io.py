# -*- coding: utf-8 -*-

import os
import errno


def create_dir(dir_name, desc='output'):
    """
    Checks if the given directory exists, if not it is created

    Returns
    -------
    dir_ok : boolean
        True if directory exists (previously or created)
        False if failed to create the directory

    dir_name : string
        Path to the directory, which may be been made absolute

    msg : string
        Error msg if directory creation failed
    """
    dir_ok = True
    if '~' in dir_name:
        dir_name = os.path.expanduser(dir_name)
    elif not os.path.isabs(dir_name):
        # Assume relative path from cwd given
        dir_name = os.path.abspath(dir_name)

    msg = "{} directory is ready".format(desc)
    errmsg = "Failed to create {} directory:\n{}\n".format(desc,
                                                           dir_name)
    if os.path.exists(dir_name):
        if os.path.isfile(dir_name):
            dir_ok = False
            errmsg += "A file already exists with the same name"
    else:
        try:
            os.makedirs(dir_name)
            msg += ("directory {} created (recursively)".format(dir_name))
        except OSError as e:
            if e.errno == errno.EEXIST:
                msg += (
                    "Assume directory {} created "
                    "(recursively) by some other process. ".format(dir_name)
                )
            else:
                dir_ok = False
                errmsg += "Underling error (makedirs) :({}) {}".format(
                    type(e).__name__, e)

    if dir_ok:
        return dir_ok, dir_name, msg
    else:
        return dir_ok, dir_name, errmsg
