import cspeed
import numpy as np

a=np.random.randint(2,4,(100,200))
b=np.random.randint(2,4,(200,200))
gpu=0
c=cspeed.matmul(a,b,gpu)


