import xml.etree.ElementTree as ET

exec_map = {}


def check(s, label=None):
    if s in exec_map:
        return exec_map[s]
    v = _check(s)
    exec_map[s] = v
    return v


# This is the oracle. Here, we use an XML parser
# but you can replace it with any context free oracle. Return
# True if your oracle agrees with the input.
def _check(s):
    try:
        ET.fromstring(s)
        return True

    except Exception:
        return False
