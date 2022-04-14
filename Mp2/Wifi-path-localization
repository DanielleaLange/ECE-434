import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from statistics import mean 
import math
import matplotlib.pyplot as plt
from scipy import stats
walk_1 = pd.read_csv('1-walk-1.csv', header=0, names=['SSID', 'Time', 'RSSI'])
walk_2 = pd.read_csv('1-walk-2.csv', header=0, names=['SSID', 'Time', 'RSSI'])
locations = pd.read_csv('ap_locations_1.csv', header=0, names=['SSID', 'x', 'y'])
offline = pd.read_csv('offline.csv', header=0, names=['x', 'y', 'alpha', 'SSID', 'RSSI'])
def walk2np(csv_thing): #input all data of a walk return 
    dim = int(csv_thing.to_numpy().shape[0]/3)
    np_thing = np.ndarray((dim, 3))

    index = [0, 0, 0]
    for w in csv_thing.to_numpy():
        if w[0] == 'a':
            np_thing[index[0]][0] = w[2]
            index[0] += 1
        elif w[0] == 'b':
            np_thing[index[1]][1] = w[2]
            index[1] += 1
        elif w[0] == 'c':
            np_thing[index[2]][2] = w[2]
            index[2] += 1
    return np_thing

def normalize(ori):
    other = np.ndarray(ori.shape)
    for col in range(ori.shape[1]):
        ori_col = abs(ori[:, col])
        other[:, col] = (ori_col - ori_col.min())/(ori_col.max() - ori_col.min())
    return other

def fit_linear(x, y):
    return stats.linregress(x, y)

def regression(ori, tx):
    reg = np.ndarray(ori.shape)
    tx = np.linspace(1, ori.shape[0], ori.shape[0])
    res_a = fit_linear(tx, ori[:, 0])
    res_b = fit_linear(tx, ori[:, 1])
    res_c = fit_linear(tx, ori[:, 2])
    reg[:, 0] = res_a.intercept + res_a.slope*tx
    reg[:, 1] = res_b.intercept + res_b.slope*tx
    reg[:, 2] = res_c.intercept + res_c.slope*tx
    return reg

def plot_wifi(norm, reg, tx, title='Walk 1'):
    plt.figure(figsize=(13, 6), dpi=144)
    plt.plot(tx, norm[:, 0], color='#78552b') # color: kyara
    plt.plot(tx, norm[:, 1], color='#58b2dc') # color: sora
    plt.plot(tx, norm[:, 2], color='#7aa23f') # color: moegi
    plt.plot(tx, reg[:, 0], ':', color='#78552b') # color: kyara
    plt.plot(tx, reg[:, 1], ':', color='#58b2dc') # color: sora
    plt.plot(tx, reg[:, 2], ':', color='#7aa23f') # color: moegi
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel('Strength (dBm)')
    plt.legend(['a', 'b', 'c'])
    plt.show()
    rint(walk_1)
    
walk_1_data = walk2np(walk_1)
tx_1 = np.linspace(1, walk_1_data.shape[0], walk_1_data.shape[0])
print(walk_1_data)

# walk_1_data_norm = normalize(walk_1_data)???? but its not normalized
walk_1_data_norm = walk_1_data

walk_1_data_reg = regression(walk_1_data_norm, tx_1)
print(walk_1_data_reg)

plot_wifi(walk_1_data_norm, walk_1_data_reg, tx_1, title='Walk 1')
#from this plot we see that the user is walking 0 degrees
#shows alpha is 0 degrees because parallel to a and b and getting closer to c 
print(walk_2)
    
walk_2_data = walk2np(walk_2)
tx_2 = np.linspace(1, walk_2_data.shape[0], walk_2_data.shape[0])
print(walk_2_data)

# walk_1_data_norm = normalize(walk_1_data)
walk_2_data_norm = walk_2_data

walk_2_data_reg = regression(walk_2_data_norm, tx_2)
print(walk_2_data_reg)

plot_wifi(walk_2_data_norm, walk_2_data_reg, tx_2, title='Walk 2')
#from this it can be determined getting closer to c and further from a and b 
#on average only angles possible are 90-0 degrees
print(offline.shape)
print(offline)
n_points = int(offline.shape[0]/3) #offline number of rows(93)/3=n_points
print(n_points) # =31 number of rows per SSID(wifi access point)
offline_data = np.ndarray((n_points, 3)) #array of size 31x3
offline_data0 = np.ndarray((9, 3)) #array of size 31x3
#offline_data90 = np.ndarray((6, 3)) #array of size 31x3
offline_data2 = np.ndarray((15, 3)) #array of size 31x3
offline_label = np.ndarray((n_points, 2)) #array of size 31x2
offline_label2 = np.ndarray((15, 2)) #array of size 31x2
offline_label0 = np.ndarray((9, 2)) #array of size 31x2
#locations/coordinates of each Ap
coor_a = (1, 3)
coor_b = (1, 1)
coor_c = (3, 2)

def e_dist(p0, p1):
    # Euclidean distance - distance between two points
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
count =0
for i in range(n_points): #traverse all rows 3 at a time to order access point 
    i3 = i*3 #this might not traverse pointes correctly???
    x = offline['x'][i3] 
    y = offline['y'][i3]
    #print(count)
    data_row2 = np.zeros((3, )) #zero temporary array 
    alpha = offline['alpha'][i3]
    #q=0
    for ap in range(3): #for 0 to 3, shouldnt it be 0 to 2??
        #set SSID and RSSI for current row
        SSID = offline['SSID'][i3+ap]
        RSSI = offline['RSSI'][i3+ap]
        #order RSSIs in order of their SSIDs     
        if SSID == 'a' and alpha == 0:
            data_row2[0] = RSSI
        elif SSID == 'b' and alpha == 0:
            data_row2[1] = RSSI
        elif SSID == 'c' and alpha == 0:
            data_row2[2] = RSSI
        elif SSID == 'a' and alpha == 90:
            data_row2[0] = RSSI
        elif SSID == 'b' and alpha == 90:
            data_row2[1] = RSSI
        elif SSID == 'c' and alpha == 90:
            data_row2[2] = RSSI
        else:
   # if alpha == 180 or alpha ==270:
            count=count+1
           # data_row2=data_row2
            break
    if alpha ==90 or alpha ==0:
        offline_data2[i-count] = data_row2
        offline_label2[i-count] = np.array([x, y])
count0=0
for i in range(n_points): #traverse all rows 3 at a time to order access point 
    i3 = i*3 #this might not traverse pointes correctly???
    x = offline['x'][i3] 
    y = offline['y'][i3]
    print(count0)
    data_row0 = np.zeros((3, )) #zero temporary array 
    alpha = offline['alpha'][i3]
    #q=0
    for ap in range(3): #for 0 to 3, shouldnt it be 0 to 2??
        #set SSID and RSSI for current row
        SSID = offline['SSID'][i3+ap]
        RSSI = offline['RSSI'][i3+ap]
        #order RSSIs in order of their SSIDs        
        if SSID == 'a' and alpha == 0:
            data_row0[0] = RSSI
        elif SSID == 'b' and alpha == 0:
            data_row0[1] = RSSI
        elif SSID == 'c' and alpha == 0:
            data_row0[2] = RSSI
        else:
           # if alpha == 180 or alpha ==270:
            count0=count0+1
           # data_row2=data_row2
            break
    if alpha ==0:
        offline_data0[i-count0] = data_row0
        offline_label0[i-count0] = np.array([x, y])
print(offline_data2)
print(offline_label2)
print(offline_label0)
print(offline_data0)
offline_tree = KDTree(offline_data0, leaf_size=40) #why 20 chosen here   In [850]:  # print(dist)
# print(ind)

def mean_point(arr):
    x = np.mean(arr[:, 0]) # mean of first column of array
    y = np.mean(arr[:, 1])# mean of second column of array 
    return x, y
    
dist_1, ind_1 = offline_tree.query(walk_1_data_reg, k=2)#?
print(dist_1)
print(ind_1)

coor_1 = []
for i in range(ind_1.shape[0]):
#     print(offline_label[ind[i]])
    x, y = mean_point(offline_label0[ind_1[i]])#should not average different phases
    print("[%.2f, %.2f]"%(x, y))
    coor_1.append([x, y])
coor_1 = np.array(coor_1)
print(coor_1)

x_min = int(coor_1[:, 0].min())
x_max = 3.5 #float(coor_1[:, 0].max())
print(x_max)
x_step = x_max - x_min + 1
tx_11 = np.linspace(x_min, x_max, coor_1.shape[0])
def regression_plot(ori, tx):#what this meant to do
    reg = np.ndarray(ori.shape)
    res_a = fit_linear(tx, ori[:, 0])
    res_b = fit_linear(tx, ori[:, 1])
    reg[:, 0] = res_a.intercept + res_a.slope*tx
    reg[:, 1] = res_b.intercept + res_b.slope*tx
    return reg
    
coor_1_reg = regression_plot(coor_1, tx_11)
print(coor_1_reg)
plt.plot(tx_11, coor_1_reg)
plt.scatter(coor_1[:, 0], coor_1[:, 1])
plt.show
offline_tree2 = KDTree(offline_data2, leaf_size=15) 
def mean_point(arr):
    x = np.mean(arr[:, 0])
    y = np.mean(arr[:, 1])
    return x, y
    

dist_2, ind_2 = offline_tree2.query(walk_2_data_reg, k=2)
print(dist_2)
print(ind_2)
coor_2 = []
for i in range(ind_2.shape[0]):
#     print(offline_label[ind_2[i]])
    x, y = mean_point(offline_label2[ind_2[i]])
    print("[%.2f, %.2f]"%(x, y))
    coor_2.append([x, y])
coor_2 = np.array(coor_2)
print(coor_2)

x_min = float(coor_2[:, 0].min())
x_max = float(coor_2[:, 0].max())
print(x_max)
print(x_min)
x_step = x_max - x_min + 1
tx_12 = np.linspace(x_min, x_max, coor_2.shape[0])
print(tx_12)
def regression_plot2(ori, tx):#what this meant to do
    reg = np.ndarray(ori.shape)
    res_a = fit_linear(tx, ori[:, 0])
    res_b = fit_linear(tx, ori[:, 1])
    reg[:, 0] = res_a.intercept + res_a.slope*tx
    reg[:, 1] = res_b.intercept + res_b.slope*tx
    return reg
    
coor_2_reg = regression_plot2(coor_2, tx_12)
print(coor_2_reg)
plt.plot(tx_12, coor_2_reg)
plt.scatter(coor_2[:, 0], coor_2[:, 1])
plt.show
