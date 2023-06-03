import pandas as pd
import numpy as np


simdata = np.load('1S2F/origin-informer/data.npz')
t = simdata['t'][:,np.newaxis]
X = simdata['X'][:,np.newaxis]
Y = simdata['Y'][:,np.newaxis]
Z = simdata['Z'][:,np.newaxis]

t = np.round(t, 3)

# combine the data into a dataframe
df1 = pd.DataFrame(np.concatenate((t, X, Y, Z), axis=1), columns=['date', 'X', 'Y', 'Z'])

# save the dataframe to a csv file
df1.to_csv('1S2F/origin-informer/1S2F_0.01.csv', index=False)
