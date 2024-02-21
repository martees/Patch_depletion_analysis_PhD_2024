import os


def is_linux():  # returns true if you're using linux, otherwise false
    try:
        test = os.uname()
        if test[0] == "Linux":
            return True
    except AttributeError:
        return False

