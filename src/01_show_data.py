import pickle
import numpy as np
import matplotlib.pyplot as plt 
## Read pickle file and get data
data=pickle.load(open('./data/rosbag.pickle','rb'))
ranges_list=data['lidar_data']
angle_increment=data['angle_increment']
fig = plt.figure()
## Using matplotlib to plot the data
show_animation = True
filtered_data=[]
for ranges in ranges_list:
    xx=[]
    yy=[]
    for i in range(len(ranges)):
        # print(ranges[i])
        
        if np.isnan(ranges[i]):
            continue
        angle=angle_increment*i
        x,y=ranges[i]*np.cos(angle),ranges[i]*np.sin(angle)
        xx.append(x)
        yy.append(y)
        np_array=np.stack((xx,yy),axis=1,dtype=np.float32)
        # print(np_array.shape)
    if show_animation:
        plt.scatter(xx,yy,c="r", marker=".") 
        plt.pause(0.1)
        plt.cla()
    filtered_data.append(np_array)
pickle.dump(filtered_data,open('./data/filtered_data.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
