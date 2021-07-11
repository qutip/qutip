import os
import qutip
import qutip.settings as qset
import qutip.solver as def_options
import warnings
try:
    import ConfigParser as configparser #py27
except:
    import configparser #py3x
from functools import partial


def getcomplex(self, section, option):
    return complex(self.get(section, option))


configparser.ConfigParser.getcomplex = getcomplex


sections = [('qutip', qset)]


def full_path(rc_file):
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
            return True, qutip_rc_file
        else:
            return False, ''
    else:
        return False, ''


def generate_qutiprc(rc_file="qutiprc"):
    """
    Generate a blank qutiprc file.
    """
    # Check for write access to home dir
    qutip_rc_file = full_path(rc_file)
    qutip_conf_dir = os.path.dirname(qutip_rc_file)
    os.makedirs(qutip_conf_dir, exist_ok=True)

    if os.path.isfile(qutip_rc_file):
        config = configparser.ConfigParser()
        config.read(qutip_rc_file)
        modified = False
        for section, settings_object in sections:
            if not config.has_section(section):
                config.add_section(section)
                modified = True
        if modified:
            with open(qutip_rc_file, 'w') as cfgfile:
                config.write(cfgfile)

        return modified

    with open(qutip_rc_file,'w') as cfgfile:
        config = configparser.ConfigParser()
        for section, settings_object in sections:
            config.add_section(section)
        config.write(cfgfile)
    return True


def get_reader(val, config):
    # The type of the read value is the same as the one presently loaded.
    if isinstance(val, bool):
        reader = config.getboolean
    elif isinstance(val, int):
        reader = config.getint
    elif isinstance(val, float):
        reader = config.getfloat
    elif isinstance(val, complex):
        reader = config.getcomplex
    elif isinstance(val, str):
        reader = config.get
    return reader


def has_rc_key(key, section=None, rc_file="qutiprc"):
    """
    Verify if key exist in section of rc_file
    """
    rc_file = full_path(rc_file)
    if not os.path.isfile(rc_file):
        return False
    config = configparser.ConfigParser()
    config.read(rc_file)
    if section is None:
        search_sections = [section for section, _ in sections]
    else:
        search_sections = [section]
    for section in search_sections:
        if config.has_section(section):
            opts = config.options(section)
            if key in opts:
                return True
    return False


def write_rc_key(key, value, section='qutip', rc_file="qutiprc"):
    """
    Writes a single key value to the qutiprc file

    Parameters
    ----------
    key : str
        The key name to be written.
    value : int/float/bool
        Value corresponding to given key.
    section : str
        String for which settings object the key belong to.
        Default : qutip
    rc_file : str
        String specifying file location.
        Default : qutiprc
    """
    rc_file = full_path(rc_file)
    if not os.access(rc_file, os.W_OK):
        warnings.warn("Does not have permissions to write config file")
        return
    config = configparser.ConfigParser()
    config.read(rc_file)
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, key, str(value))

    with open(rc_file, 'w') as cfgfile:
        config.write(cfgfile)


def read_rc_key(key, datatype, section='qutip', rc_file="qutiprc"):
    """
    Writes a single key value to the qutiprc file

    Parameters
    ----------
    key : str
        The key name to be written.
    datatype :
        Type of the value corresponding to given key.
        One of [int, float, bool, complex, str]
    section : str
        String for which settings object the key belong to.
    rc_file : str
        String specifying file location.
    """
    rc_file = full_path(rc_file)
    config = configparser.ConfigParser()
    config.read(rc_file)
    if not config.has_section(section):
        raise ValueError("key not found")
    reader = get_reader(datatype(0), config)
    opts = config.options(section)
    if key not in opts:
        raise ValueError("key not found")
    return reader(section, key)


def write_rc_object(rc_file, section, object):
    """
    Writes all keys and values corresponding to one object qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    section : str
        Tags for the saved data.
    object : Object
        Object to save. All attribute's type must be one of bool, int, float,
        complex, str.
    """
    generate_qutiprc(rc_file)
    config = configparser.ConfigParser()
    config.read(full_path(rc_file))
    if not config.has_section(section):
        config.add_section(section)
    keys = object.__all
    for key in keys:
        config.set(section, key, str(getattr(object, key)))
    with open(full_path(rc_file), 'w') as cfgfile:
        config.write(cfgfile)
    return


def load_rc_object(rc_file, section, object):
    """
    Read keys and values corresponding to one settings location
    to the qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    section : str
        Tags for the saved data.
    object : Object
        Object to overwrite. All attribute's type must be one of bool, int,
        float, complex, str.
    """
    config = configparser.ConfigParser()
    config.read(full_path(rc_file))
    if not config.has_section(section):
        raise configparser.NoSectionError(section)
    keys = object.__all
    opts = config.options(section)
    for op in opts:
        if op in keys:
            reader = get_reader(getattr(object, op), config)
            setattr(object, op, reader(section, op))
        else:
            warnings.warn("Invalid qutip config variable in qutiprc: " + op)


def write_rc_qset(rc_file):
    """
    Writes qutip.settings in a qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    """
    write_rc_object(rc_file, "qutip", qset)


def load_rc_qset(rc_file):
    """
    Read qutip.settings to a qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    """
    load_rc_object(rc_file, "qutip", qset)


def write_rc_config(rc_file):
    """
    Writes all keys and values to the qutiprc file.

    Parameters
    ----------
    rc_file : str
        String specifying file location.
    """
    generate_qutiprc(rc_file)

    config = configparser.ConfigParser()
    config.read(full_path(rc_file))
    for section, settings_object in sections:
        keys = settings_object.__all
        for key in keys:
            config.set(section, key, str(getattr(settings_object, key)))

    with open(full_path(rc_file), 'w') as cfgfile:
        config.write(cfgfile)
    return


def load_rc_config(rc_file):
    """
    Loads the configuration data from the
    qutiprc file
    """
    config = configparser.ConfigParser()
    config.read(full_path(rc_file))
    for section, settings_object in sections:
        if config.has_section(section):
            keys = settings_object.__all
            opts = config.options(section)
            for op in opts:
                if op in keys:
                    reader = get_reader(getattr(settings_object, op), config)
                    setattr(settings_object, op,
                            reader(section, op))
                else:
                    warnings.warn("Invalid qutip config variable in qutiprc: "
                                  + op)
                    # raise Exception('Invalid config variable in qutiprc.')
        else:
            warnings.warn("Section " + section + " not found ")
            # raise configparser.NoSectionError('qutip')

    if config.has_section('compiler'):
        _valid_keys = ['CC', 'CXX']
        opts = config.options('compiler')
        for op in opts:
            up_op = op.upper()
            if up_op in _valid_keys:
                os.environ[up_op] = config.get('compiler', op)
            else:
                # raise Exception('Invalid config variable in qutiprc.')
                warnings.warn("Invalid compiler config variable in qutiprc: "
                              + op)
