unit_suffix = {
    "K":   1e3, "k":   1e3,
    "MEG": 1e6, "meg": 1e6,
    "G":   1e9, "g":   1e9, "giga":   1e9,
    "T":  1e12, "t":  1e12, "terra": 1e12,

    "M":  1e-3, "m":  1e-3, "milli":  1e-3,
    "U":  1e-6, "u":  1e-6, "micro":  1e-6,
    "N":  1e-9, "n":  1e-9, "nano":   1e-9,
    "P": 1e-12, "p": 1e-12, "pico":  1e-12,
    "F": 1e-15, "f": 1e-15, "femto": 1e-15
}

def parse_ltspice_number(number):
    for i, x in enumerate(number):
        if x.isalpha():
            return float(number[:i]) * unit_suffix[number[i:]]

    return float(number)

def parse_ltspice_txt(path):
    data = []
    start_token = "Step Information: "
    var_names = None

    with open(path) as f:
        var1, var2 = f.readline().rstrip().split()

        for line in f:
            line = line.rstrip() 
            if line.startswith(start_token):
                parts = [x.split("=") for x in line.split() if "=" in x]
                
                var_names = [x[0] for x in parts]
                var_values = [parse_ltspice_number(x[1]) for x in parts]

            else:
                var1_value, var2_value = [float(x) for x in line.split()]
                data_point = {
                    var1: var1_value,
                    var2: var2_value,
                }
                if var_names is not None:
                    for name, value in zip(var_names, var_values):
                        data_point[name] = value
                
                data.append(data_point)
    
    return data

def calculate_line(x0: float, y0: float, slope: float, xs=None, ys=None):
    if xs is None and ys is None:
        raise ValueError

    if xs is not None:
        return y0 + (xs - x0) * slope

    return x0 + (ys - y0) / slope
