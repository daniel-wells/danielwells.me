import pandas as pd
import plotnine as pn
import numpy as np

def percent_format2(x):
    labels = [f"{v*100}%" for v in x]
    pattern = re.compile(r'\.0+%$') #
    labels = [pattern.sub('%', val) for val in labels]
    return labels

def concentration(t, ach=2, k=1.5):
    """
    t: time in hours
    ach: Air Changes per Hour
    k: mixing factor (2 meaning 50% mixing)
    Assumes perfect mixing and no continous addition/release of particles
    """
    return np.exp(-1 * t * ach * 1/k)

decay = []
for ach in [1,2,3,4,5,6]:
    tmp = pd.DataFrame({'time': np.linspace(0,2.2,100) * 60,
                        'particles': concentration(np.linspace(0,2.2,100), ach),
                        'ACH': ach})
    decay.append(tmp)

decay = pd.concat(decay)

p = pn.ggplot(decay, pn.aes('time', 'particles', colour='ACH', group='ACH')) +\
        pn.ylab('Particles remaining') +\
        pn.xlab('Time (Minutes)') +\
        pn.geom_line() +\
        pn.coord_cartesian(xlim=[0,120]) +\
        pn.scale_x_continuous(breaks=15 * np.arange(0,9,1)) +\
        pn.scale_color_gradient2(low='black', mid='#feb24c', high='#f03b20', midpoint=3.5)

(p + pn.scale_y_continuous(labels=percent_format2)).save("images/decay_linear.png", dpi=200)
(p + pn.scale_y_log10(labels=percent_format2)).save("images/decay_log.png", dpi=200)
