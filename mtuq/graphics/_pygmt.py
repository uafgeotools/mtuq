
def exists_pygmt():
    try:
        import pygmt
        return True
    except:
        return False


