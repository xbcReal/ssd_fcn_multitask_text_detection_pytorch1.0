import numpy as np
a=np.zeros(shape=(2,8))
b=np.zeros(shape=(100,8))
a=np.array([[1],[2]])
b=np.array([[1],[2],[3]])
ret=[]
for i in range(np.shape(a)[0]):
    # ret_temp=b-a[i,:].sum(axis=1,keepdims=True)
    ret_temp = np.sum(np.abs(b - a[i, :]),axis=1,keepdims=True)
    ret.append(ret_temp)
ret=np.concatenate(ret,axis=1)
print(np.shape(ret))
print(ret)