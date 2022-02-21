import numpy as np
import pandas as pd


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
    sweep_var_names = None

    with open(path) as f:
        var_names = f.readline().rstrip().split()

        for line in f:
            line = line.rstrip() 
            if line.startswith(start_token):
                parts = [x.split("=") for x in line.split() if "=" in x]
                
                sweep_var_names = [x[0] for x in parts]
                sweep_var_values = [parse_ltspice_number(x[1]) for x in parts]

            else:
                var_values = [float(x) for x in line.split()]
                data_point = {name: value for name, value in zip(var_names, var_values)}
                if sweep_var_names is not None:
                    for name, value in zip(sweep_var_names, sweep_var_values):
                        data_point[name] = value
                
                data.append(data_point)
    
    return data


def parse_ltspice_raw(path, ascii=False):
    return parse_ltspice_raw_ascii(path) if ascii else parse_ltspice_raw_bin(path)


def parse_ltspice_raw_ascii(path):
    variables = []
    values = []
    with open(path) as f:
        # seek to line "Variables:"
        while True:
            line = f.readline().rstrip()
            if line == "Variables:":
                break

        # read variables' names
        while True:
            line = f.readline().rstrip()
            if line == "Values:":
                break
            variables.append(line.split()[1])

        # read values
        while True:
            line = f.readline().rstrip()
            if line:
                new_values = [line.split()[-1]]
                for _ in range(len(variables)-1):
                    new_values.append(f.readline().rstrip().split()[-1])
                new_values = [float(x) for x in new_values]
                values.append(new_values)
            else:
                break
        
    return variables, values


def parse_ltspice_raw_bin(path):
    with open(path, 'rb') as f:
        bin_data = f.read()

    var_names = []
    start_append = False
    curr_i = 0
    num_vars = 0
    for i in range(len(bin_data)):
        if bin_data[i:i+1] == b'\n':        # new line
            line = bin_data[curr_i:i].decode('utf-16')
            curr_i = i + 2      # skip '\n'
            
            if line == 'Binary:':
                break
            
            if line.startswith('No. Variables: '):
                num_vars = int(line[len('No. Variables: '):])
            
            if start_append:
                var_names.append(line.split('\t')[2])
            
            if line == 'Variables:':
                start_append = True

    data = np.frombuffer(bin_data[curr_i:], dtype=np.float64).reshape(-1, num_vars)
    df = pd.DataFrame(data, columns=var_names)
    return df


def calculate_line(x0: float, y0: float, slope: float, xs=None, ys=None):
    if xs is None and ys is None:
        raise ValueError

    if xs is not None:
        return y0 + (xs - x0) * slope

    return x0 + (ys - y0) / slope
