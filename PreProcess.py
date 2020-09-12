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
def combine_npy(dir1,dir2):
    data1=np.load(dir1,allow_pickle=True)
    data2=np.load(dir2,allow_pickle=True)
    specialsub=data1[2]
    # data2=np.squeeze(data2,axis=0)
    # data2=np.array(data2)
    specialsub=np.array(specialsub).reshape(1,-1)
    print(data2.shape,specialsub.shape)
    newsub=np.concatenate((data2,specialsub),1).reshape(-1)
    newsub=newsub.tolist()
    print(len(newsub))
    newdata=[]
    for i in range(data1.shape[0]):
        if i==2:
            newdata.append(newsub)
        else:
            newdata.append(data1[i])
    newdata=np.array(newdata)
    # print(newdata[2].shape)
    # ipdb.set_trace()
    np.save('label3.npy',newdata)

def load_mat_to_npy(dir):
    data=os.listdir(dir)
    totalEcgData=[]
    totalGsrData=[]
    totallabel=[]
    for sub in data:
        filedir=dir+'/'+str(sub)
        print(filedir)
        data=loadmat(filedir)
        ecg=data['ecg_segment']
        gsr=data['gsr_segment']
        label=data['label']
        label=label.reshape(-1)
        totalEcgData.append(ecg)
        totalGsrData.append(gsr)
        totallabel.append(label)
    # ipdb.set_trace()
    np.save('Ecg.npy',np.array(totalEcgData))
    np.save('EcgGsrLabel.npy',np.array(totallabel))
    np.save('Gsr.npy',np.array(totalGsrData))
    # print(data)

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
        # print(sample[1]['comments'])
        # print(sample[0][:,1].shape)
        ECG.append(sample[0][:,1])
        ECG.append(sample[0][:,0])
        # plt.plot(sample[0][1:1000,1])
        # plt.show()
        # break
    ECG = np.asarray(ECG)
    return ECG
def load_ppg(FilePath):
    originalData=loadmat(FilePath)
    # print(originalData)
    ppg=originalData['ppg']
    return  ppg
    # print(ppg.shape)
def load_ecg(filepath):
    # judge file suffix
    if filepath.split('/')[-1][-4:]=='.mat':
        signal=loadmat(filepath)
        signal=signal['ecg']
    else:
        ecgsignal=np.load(filepath,allow_pickle=True)
        # totalsig=[]
        for i in range(ecgsignal.shape[0]):
            # print(ecgsignal[i].shape)
            if i==0:
                signal =ecgsignal[i]
            else:
                signal=np.concatenate((signal, ecgsignal[i]), axis=0)
            # totalsig.append(ecgsignal[i])
        # totalsig=np.array(totalsig)
        # print(signal.shape)
        # print('ecgsignal shape:',ecgsignal.shape)
        # print('type ecgsignal:',type(ecgsignal))
        # print(ecgsignal[1].shape)

        #show the clean ecg data
        # signal=signal.reshape(1,-1)
        # for i in range(5):
        #     plt.plot(signal[0][i*125:(i+1)*125],label='clean')
        #     plt.legend()
        #     plt.show()
    return signal
def load_noisy(FilePath):
    originaldata=wfdb.rdsamp(FilePath)
    noisydata=originaldata[0]
    # print(noisydata.shape)
    return noisydata
def load_ecgsegment(FilePath):
    originalData = loadmat(FilePath)
    # print(originalData)
    ecg = originalData['ecg_segment']
    ecg=ecg.reshape(1,-1)
    return ecg
    # print(ppg.shape)
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
    load_mat_to_npy('/home/wcj/CurrentProject/ECGDenoising/OwnEcgGsrData')
    # combine_npy('/home/wcj/CurrentProject/EmotionRecongntion/dreamer-data/label1.npy','/home/wcj/CurrentProject/EmotionRecongntion/dreamer-data/label_xgc.npy')

    '''
    ecgpath = '/home/wcj/CurrentProject/EmotionRecongntion/dreamer-data/data1.npy'
    load_ecg(ecgpath)
    a=np.array([[1,2,3]])
    b=np.array([[4,5,7]])
    print(a.shape)
    print(b.shape)
    c=np.concatenate((a,b),axis=0)
    print(c.shape)
    c=c.reshape(-1)
    print(c)
    '''

    '''

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

    
    # load_ppg('/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg.mat')
    # load_noisy('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/em')
    '''

    # load mit ecg database data and save to mat file
    # ecg=load_data('/home/wcj/DataSet/physionet.org/files/mitdb/1.0.0')
    # ecg=ecg.reshape(1,-1)
    # print(ecg.shape)
    # savemat('mit_oriecg.mat',{'ecg':ecg})
    # plt.plot(ecg[0,3000:4000])
    # plt.show()

    # original_data=wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/118e06')
    # original_data = wfdb.rdsamp('/home/wcj/DataSet/physionet.org/files/mitdb/1.0.0/222')
    # print(original_data[0].shape)
    # plt.plot(original_data[0][1:1000,0])
    # plt.show()
    # print(original_data[1]['comments'])
    # print(type(original_data[1]['comments']))
    # print(len(original_data[1]['comments']))
    # print(original_data[1]['comments'][0].split(' ')[-2])

    '''
    test code
    '''
    # filepath='/home/wcj/CurrentProject/ECGDenoising/mit_oriecg.mat'
    # print(filepath.split('/')[-1][-4:])


    # b=[]
    # for i in range(20):
    #     b.append(i)
    # b=np.array(b).reshape(1,-1)
    # for j in range(5):
    #     print(b[0,j*4:(j+1)*4])