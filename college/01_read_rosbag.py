import embag
import struct
import numpy as np
import struct
view = embag.View().addBag('./data/college/rooster_2020-07-10-09-13-52_0-003.bag')
import matplotlib.pyplot as plt
count = 0
lidar_data=[]
save_dict={}
struct_fmt = f'<{1024*64*12}f'
struct_unpack = struct.Struct(struct_fmt).unpack_from
output_data=[]

for msg in view.getMessages(['/os1_cloud_node/points']):
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')   
    # print(msg.timestamp.to_sec())
    msg_dict=msg.dict()
    # print(msg_dict.keys())
    # print(msg_dict['row_step'])
    # print(msg_dict['width'])
    # print(msg_dict['fields'])
    # print(len(msg_dict['data']))
    # np.fromstring()
    data=struct_unpack(msg_dict['data'])
    data=np.asarray(data).reshape(1024,64,12)
    data=data[:,0,0:3]

    # data=np
    # for i in range(1024*64):
    output_data.append(data)    
    # data=np.asarray(data).reshape(64*1024,3)
    # plt.scatter(data[:,0],data[:,1],marker='.')
    # output_data.append(data[0,:,0:2])
    # plt.show()
    # data=msg_dict['buf']
    count+=1
    # if count==50:
    #     break
    # print(msg)
import pickle
pickle.dump(output_data,open('./data/college/2dicp.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)




