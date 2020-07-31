import wfdb
import matplotlib.pyplot as plt
import os
import numpy as np
import ipdb
from scipy.io import loadmat,savemat
import math
from sklearn import  preprocessing
plt.rcParams['figure.figsize']=(30,10)

# load data from nstdb
# *********noisy type**********
# ma:muscle (EMG) artifact (in record 'ma')
# bw:baseline wander (in record 'bw')
# em:electrode motion artifact (in record 'em')
# *****link**********
# https://physionet.org/content/nstdb/1.0.0/
def load_data(fileDir):
    file_path_list = []
    valid_file_extensions = [".dat"]
    valid_file_extensions = [item.lower() for item in valid_file_extensions]

    for file in os.listdir(fileDir):
        extension = os.path.splitext(file)[1]
        # print(extension)
        if extension.lower() not in valid_file_extensions:
            continue
        file_path_list.append(os.path.join(fileDir, file))
    print(len(file_path_list))
    ECG = []
    for path in file_path_list:
        base=os.path.basename(path)
        base = os.path.splitext(base)[0]
        print(fileDir+'/%s'%(base))
        sample = wfdb.rdsamp(fileDir+'/%s'%(base))
        print(sample[1]['comments'])
        ECG.append(sample[0][:,1])
    ECG = np.asarray(ECG)
    return ECG
def load_ppg(FilePath):
    originalData=loadmat(FilePath)
    # print(originalData)
    ppg=originalData['ppg']
    return  ppg
    # print(ppg.shape)
def load_noisy(FilePath):
    originaldata=wfdb.rdsamp(FilePath)
    noisydata=originaldata[0]
    # print(noisydata.shape)
    return noisydata

def calculategain(snr,ppg,noisy):
    '''
    
    :param snr: 
    :param ppg: 
    :param noisy: 
    :return: gain(增益)
    :according to the given snr to generate all kinds of noisy level signal
    :reference snr=10*log(S/(N*a*a))
    '''
    # return math.sqrt(np.sum(ppg**2)/(np.sum(ppg**2)/(math.pow(10,snr/10)*np.sum(noisy**2))))
    return math.sqrt(np.sum(ppg**2)/(math.pow(10,snr/10)*np.sum(noisy**2)))

def generate_signal(gain,clean,noisy):
    '''
    :return:
    gain*preprocessing.scale(noisy)
    '''
    # noisydata = preprocessing.scale(clean) + gain*preprocessing.scale(noisy)
    # noisydata=preprocessing.scale(clean)+gain*noisy
    noisydata = clean + gain * noisy
    return noisydata

if __name__ == '__main__':
    # load_data('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0')

    # original_data=wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/118e06')
    # print(original_data)
    # print(original_data[1]['comments'])
    # print(type(original_data[1]['comments']))
    # print(len(original_data[1]['comments']))
    # print(original_data[1]['comments'][0].split(' ')[-2])

    # show noisy data
    em_data = wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/em')
    bw_data = wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/bw')
    ma_data = wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/ma')
    print("em_data:", em_data)
    print("bw_data:", bw_data)
    print("ma_data:", ma_data)
    plt.plot(em_data[0][1:1000, 1], label='em')
    plt.plot(bw_data[0][1:1000, 1], label='bw')
    plt.plot(ma_data[0][1:1000, 1], label='ma')
    plt.legend()
    plt.show()

    # read clean ecg
    data118 = wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/mitdb/1.0.0/118')
    # data119=wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/mitdb/1.0.0/119')
    # original_data=wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/118e00')
    # print(data118)
    # plt.plot(data118[0][1:1000,1],'r-',label='118e00')
    plt.plot(data118[0][1:1000, 1], label='118')
    # plt.plot(data119[0][1:1000,1],label='119')
    plt.legend()
    plt.show()

    # load_ppg('/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg.mat')
    # load_noisy('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/em')
