def get_pass(var: str):
    import os
    import getpass
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")
