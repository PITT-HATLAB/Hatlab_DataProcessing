def t2f(t_unit):
    if t_unit == "ns":
        return "GHz"
    elif t_unit == "us":
        return "MHz"
    elif t_unit == "ms":
        return "kHz"
    elif t_unit == "s":
        return "Hz"
    else:
        raise NameError("unsupported time unit")

def f2t(f_unit):
    if f_unit == "GHz":
        return "ns"
    elif f_unit == "MHz":
        return "us"
    elif f_unit == "kHz":
        return "ms"
    elif f_unit == "Hz":
        return "s"
    else:
        raise NameError("unsupported freq unit")