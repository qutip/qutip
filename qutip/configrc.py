import os
import warnings
from configparser import (ConfigParser, MissingSectionHeaderError,
                          ParsingError, NoSectionError)
from functools import partial
import pickle


getter = [bool, int, float, complex, str]


def _get_str(value):
    if type(value) in getter:
        return str(value)
    return pickle.dumps(value).hex()


def _get_reader(datatype):
    if datatype is bool:
        return lambda bool_str: bool_str == "True"
    if datatype in getter:
        return datatype
    return lambda x: pickle.loads(bytes.fromhex(x))


def _full_path(rc_file):
    rc_file = os.path.expanduser(rc_file)
    if os.path.isabs(rc_file):
        return rc_file
    qutip_conf_dir = os.path.join(os.path.expanduser("~"), '.qutip')
    return os.path.join(qutip_conf_dir, rc_file)


def has_qutip_rc():
    """
    Checks to see if the qutiprc file exists in the default
    location, i.e. HOME/.qutip/qutiprc
    """
    qutip_conf_dir = os.path.join(os.path.expanduser("~"), '.qutip')
    if os.path.exists(qutip_conf_dir):
        qutip_rc_file = os.path.join(qutip_conf_dir, "qutiprc")
        qrc_exists = os.path.isfile(qutip_rc_file)
        if qrc_exists:
            return True
        else:
            return False
    else:
        return False


def generate_qutiprc(rc_file="qutiprc"):
    """
    Generate a blank qutiprc file.
    """
    # Check for write access to home dir
    qutip_rc_file = _full_path(rc_file)
    qutip_conf_dir = os.path.dirname(qutip_rc_file)
    os.makedirs(qutip_conf_dir, exist_ok=True)

    if os.path.isfile(qutip_rc_file):
        try:
            config = ConfigParser()
            config.read(qutip_rc_file)
        except (MissingSectionHeaderError, ParsingError):
            # Not a valid file, overwrite
            pass
        else:
            if "qutip" in config:
                return

    with open(qutip_rc_file, 'w') as cfgfile:
        config = ConfigParser()
        config["qutip"] = {}
        config.write(cfgfile)


def has_rc_key(rc_file, key, section=None):
    """
    Verify if key exist in section of rc_file
    """
    rc_file = _full_path(rc_file)
    try:
        config = ConfigParser()
        config.read(rc_file)
    except (MissingSectionHeaderError, ParsingError):
        return False
    if section is not None:
        return section in config and key in config[section]
    else:
        return any(key in section for section in config)


def write_rc_key(rc_file, key, value, section):
    """
    Writes a single key value to the qutiprc file

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    key : str
        The key name to be written.
    value : int/float/bool
        Value corresponding to given key.
    section : str
        String for which settings object the key belong to.
    """
    config = ConfigParser()
    config.read(_full_path(rc_file))
    if section not in config:
        config[section] = {}
    config[section][key] = _get_str(value)

    with open(rc_file, 'w') as cfgfile:
        config.write(cfgfile)


def read_rc_key(rc_file, key, datatype, section):
    """
    Writes a single key value to the qutiprc file

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    key : str
        The key name to be written.
    datatype :
        Type of the value corresponding to given key.
        One of [int, float, bool, complex, str]
    section : str
        String for which settings object the key belong to.
    """
    config = ConfigParser()
    config.read(_full_path(rc_file))
    reader = _get_reader(datatype)
    return reader(config[section][key])


def has_rc_object(rc_file, name):
    """
    Read keys and values corresponding to one settings location
    to the qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    section : str
        Tags for the saved data.
    """
    config = ConfigParser()
    try:
        config.read(_full_path(rc_file))
    except (MissingSectionHeaderError, ParsingError):
        return False
    return section in config


def write_rc_object(rc_file, objs):
    """
    Writes all keys and values corresponding to one optionsclass to a
    qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    section : str
        Tags for the saved data.
    objs : list of optionsclass
        Object to save. Must be decorated with `optionsclass`.
    """
    generate_qutiprc(rc_file)
    config = ConfigParser()
    config.read(_full_path(rc_file))
    for obj in objs:
        config[obj._name] = {}
        for key, value in obj.options.items():
            config[obj._name][key] = _get_str(value)
    with open(_full_path(rc_file), 'w') as cfgfile:
        config.write(cfgfile)


def load_rc_object(rc_file, objs):
    """
    Read keys and values corresponding to one settings location
    to the qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    obj : list of optionsclass
        Object to overwrite.
    """
    config = ConfigParser()
    config.read(_full_path(rc_file))
    for obj in objs:
        if obj._name in config:
            _load_one_obj(config, obj)


def _load_one_obj(config, obj):
    for option in config[obj._name]:
        if option in obj.options:
            reader = _get_reader(obj._types[option])
            obj.options[option] = reader(config[obj._name][option])
