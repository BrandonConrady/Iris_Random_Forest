# misc. methods taken out of main.py to keep things a little clean

def to_percent(value: float) -> str:
    percent = value * 100

    return ("%.3f" % percent) + "%"