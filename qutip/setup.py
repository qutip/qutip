import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('qutip', parent_package, top_path)
    # Add all QuTiP subpackages here:
    config.add_subpackage('cy')
    config.add_subpackage('control')
    if os.environ['FORTRAN_LIBS'] == 'TRUE':
        config.add_subpackage('fortran')

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
