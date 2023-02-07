import embag
import struct
import numpy as np
view = embag.View().addBag('/Users/hu/Downloads/rosbag.bag')
count = 0
lidar_data=[]
save_dict={}
for msg in view.getMessages(['/LiDAR/LD19']):
    # print(msg.timestamp.to_sec())
    # print(msg)
    
    
    # There are a few ways to access fields, the first returns a dict of the message
    msg_dict=msg.dict()
    if count==0:
        save_dict['angle_increment']=msg_dict['angle_increment']
        save_dict['angle_max']=msg_dict['angle_max']
    
    ranges=msg_dict['ranges']
    # print(len(ranges))
    lidar_data.append(ranges)
    intensity=msg_dict['intensities']
    # a=input()
    # You can also access individual fields much faster this way
    # print(msg.data()['cool_field'])

    # Arrays are returned as Python bytes for performance reasons.
    # Assuming an array of floats (4 bytes per value), you can unpack them using:
    # values = struct.unpack('<%df' % (len(msg.data()['field']) / 4), msg.data()['field'])
save_dict['lidar_data']=lidar_data

import pickle
pickle.dump(save_dict,open('rosbag.pickle','wb'),pickle.HIGHEST_PROTOCOL)