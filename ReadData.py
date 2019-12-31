from numpy import load


#Get the Camera Matrix
data = load('Camera_Param.npz') 
#data.files >>> ['CameraMtx', 'DistVector']
K = data['CameraMtx'] 
d = data['DistVector']

