from numpy import load

#Get the Camera Matrix and distortion coefficients
def read_calibration():
    Calib_data = load('Calibration.npz') 
    #Calib_data.files >>> ['CameraMtx', 'DistortionVec']
    K = Calib_data['CameraMtx'] 
    d = Calib_data['DistortionVec']
    return K, d


 #Get the Rotaion Matrix and Translation vector
def read_alignmnet():
    Align_data = load('Alignment.npz') 
    #Align_data.files >>> ['RotationVec', 'TranslationVec']
    R = Align_data['RotationMtx']
    t = Align_data['TranslationVec']
    return R, t
