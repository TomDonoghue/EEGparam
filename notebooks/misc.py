def all_areas(dat):
    outs = []
    for dd in dat:
        outs.append(area_gauss(dd[1], dd[2]))
    return np.array(outs)

def area_gauss(amp, bw):
    return (amp * bw/2) / 0.3989
