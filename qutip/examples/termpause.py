import sys

def termpause():
    """
    Pause's script running in a terminal window until the user presses a key,
        else does nothing.
    """
    if sys.stdout.isatty():
        uinpt=raw_input("\nPress any key to continue...\n")
        return
    else:
        print ''
        pass


