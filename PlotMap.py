import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 

city_name = []
city_condition = []
with open('data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        city_name.append(line[0])
        city_condition.append([float(line[1]), float(line[2])])
city_condition = np.array(city_condition)

result_path = [15, 13, 25, 26, 0, 27, 21, 29, 30, 23, 5, 28, 1, 2, 11, 24, 3, 4, 10, 9, 6, 7, 8, 12, 14, 16, 22, 17, 18, 19, 20, 15]


X = []
Y = []
for index in result_path:
    X.append(city_condition[index, 0])
    Y.append(city_condition[index, 1])

fig = plt.figure(figsize=(20,16))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
map = Basemap(projection='poly',lat_0=35,lon_0=110,llcrnrlon=80,llcrnrlat=3.01,urcrnrlon=140,urcrnrlat=53.123,resolution='h',area_thresh=1000,rsphere=6371200.,ax = ax1)
map.readshapefile("./template/bou2_4p","china",drawbounds=True)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'coral',alpha = .1)
map.drawmapboundary()
map.drawparallels(np.arange(0.,90,10.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(80.,140.,10.),labels=[0,0,0,1],fontsize=10)


for i in list(range(0,31)):
    if i == 31:
        start_lon = X[31]
        start_lat = Y[31]
        end_lon = X[0]
        end_lat = Y[0]
    else:
        start_lon = X[i]
        start_lat = Y[i]
        end_lon = X[i+1]
        end_lat = Y[i+1]
    if abs(end_lat - start_lat) < 180 and abs(end_lon - start_lon) < 180:
        map.drawgreatcircle(start_lon, start_lat, end_lon, end_lat, linewidth=1, color = "red")



x, y = map(X, Y)
map.scatter(x, y, marker='o', color='m')
plt.savefig('./Result.png')
plt.show()

