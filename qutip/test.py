from qobj import * 
from states import *
import numpy as np 
not_dm = Qobj(np.random.rand(4,4)).unit()
print(not_dm.isherm)
print(not_dm.tr())
print(not_dm.eigenenergies())
print(not_dm.purity())
