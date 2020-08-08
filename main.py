from  model import  *
from PreProcess import  *

# TrainNoisySet=[]
# ValNoisySet=[]
# TestNoisytSet=[]
#
# TrainCleanSet=[]
# ValCleanSet=[]
# TestCleanSet=[]


def save_slicesignal():
    ppgsignal=[]
    ppgcleanfrag=[]
    ppgnoisyfrag={}
    noisyfrag=[]
    SignalLength=1000

    ppgpath1 = '/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg.mat'
    ppgpath2 = '/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg1.mat'
    empath = '/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/em'
    bwpath = '/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/bw'
    mapath = '/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/ma'

    ppgsignal1 = load_ppg(ppgpath1)
    ppgsignal2 = load_ppg(ppgpath2)
    # ppgsignal1=np.reshape(ppgsignal1,(-1))
    # ppgsignal2=np.reshape(ppgsignal2,(-1))
    em_data = load_noisy(empath)
    bw_data = load_noisy(bwpath)
    ma_data = load_noisy(mapath)
    em_data = em_data.T
    bw_data = bw_data.T
    ma_data = ma_data.T

    print(ppgsignal1.shape, ppgsignal2.shape, bw_data.shape)
    ppgsignal = np.concatenate((ppgsignal1, ppgsignal2), axis=1)
    # ppgsignal.append(ppgsignal1)
    # ppgsignal.append(ppgsignal2)
    # ppgsignal=ppgsignal1+ppgsignal2
    # ppgsignal=np.array(ppgsignal)
    # print(ppgsignal.shape)
    print(ppgsignal.shape)

    def slice_signal(Clean, Noisy, length):
        CleanLength = Clean.shape[1]
        NoisyLength = Noisy.shape[1]
        # Clean=preprocessing.scale(Clean)
        # print(Clean.shape)
        TotalLength = min(CleanLength, NoisyLength)

        '''
        snr
        '''
        # for j in range(0, 60, 5):
        for j in range(0, 20, 5):
            for i in range(0, TotalLength - length, length):
                # print(i)
                # for i in range(1):
                fragppg = Clean[0, i:i + 1000].reshape(-1)
                fragppg = preprocessing.scale(fragppg)
                fragnoisy = Noisy[0, i:i + 1000].reshape(-1)
                ppgcleanfrag.append(fragppg)
                noisyfrag.append(fragnoisy)
                '''
                calculate gain
                '''
                gain = calculategain(j, fragppg, fragnoisy)
                # print("gain:",gain)
                # gain=1
                key = str(j) + "dB"

                if i != 0:
                    ppgnoisyfrag[key].append(generate_signal(gain, fragppg, fragnoisy))
                else:
                    ppgnoisyfrag[key] = [generate_signal(gain, fragppg, fragnoisy)]

    '''
    save clean,noisy,synthetic signal
    '''

    TotalNoisyData = {}
    TotalNoisyData['ma'] = ma_data
    TotalNoisyData['bw'] = bw_data
    TotalNoisyData['em'] = em_data
    TypeNoisy = ['ma', 'bw', 'em']
    for i, type in enumerate(TypeNoisy):
        # ppgcleanfrag = []
        ppgnoisyfrag = {}
        noisyfrag = []
        slice_signal(ppgsignal, TotalNoisyData[type], SignalLength)
        # print(np.array(ppgcleanfrag).shape)
        noisyname = ("{}.mat".format(type))
        savemat(noisyname, {"sig": np.array(noisyfrag)})
        for j in range(0, 20, 5):
            key = str(j) + "dB"
            # print(type, key)
            syntheticname = ("ppg_{}_{}".format(type, key))
            savemat(syntheticname, {"sig": np.array(ppgnoisyfrag[key])})
    cleanname = ("ppg_clean.mat")
    savemat(cleanname, {"sig": np.array(ppgcleanfrag)})





# slice_signal(ppgsignal,em_data,1000)
# print(np.array(ppgcleanfrag).shape)
# a=np.array([1,3,5,6])
# b=a[0:4]
# print(type(b),b.shape)
# print("b:",b)
# for j in range(0, 60, 5):

'''
plot the different snr noisy signal
'''
# for j in range(0,20,5):
#     key=str(j)+"dB"
#     print(key)
#     tempsig=np.array(ppgnoisyfrag[key])
#     for k in range(tempsig.shape[0]):
#         # plt.plot(ppgcleanfrag[k],label='original')
#         plt.plot(ppgcleanfrag[k],label='scale')
#         plt.plot(tempsig[k],label='noisy')
#         plt.title("snr:{},noisy:em".format(key))
#         plt.legend()
#         plt.show()
#         break


def GetDataSet():
    total_noisy_ppg=[]
    clean_ppg_path='/home/wcj/CurrentProject/ECGDenoising/ppg_clean.mat';
    original_data=loadmat(clean_ppg_path);
    clean_ppg=original_data['sig']
    print('clean ppg shape:',clean_ppg.shape)

    TypeNoisy = ['ma', 'bw', 'em']
    for i, type in enumerate(TypeNoisy):
        for j in range(0, 20, 5):
            key = str(j) + "dB"
            # print(type, key)
            noisy_ppg_path='/home/wcj/CurrentProject/ECGDenoising/ppg_'+str(type)+'_'+key+'.mat'
            original_data = loadmat(noisy_ppg_path);
            noisy_ppg = original_data['sig']
            total_noisy_ppg.append(noisy_ppg)
    total_noisy_ppg=np.array(total_noisy_ppg).reshape(-1,1000)
    print('noisy ppg shape',total_noisy_ppg.shape)



    '''
    split the noisy and respond clean data to make train data
    '''
    total_count=clean_ppg.shape[0]
    # print(total_count)
    index1=int(total_count*(2/3))
    index2=int(total_count*(5/6))
    # print(index1,index2)
    TrainCleanSet=clean_ppg[0:index1][:]
    TrainNoisySet=total_noisy_ppg[0:index1][:]

    ValCleanSet=clean_ppg[index1:index2][:]
    ValNoisySet=total_noisy_ppg[index1:index2][:]

    TestCleanSet=clean_ppg[index2:total_count][:]
    TestNoisytSet=total_noisy_ppg[index2:total_count][:]

    print('TrainCleanSet:',TrainCleanSet.shape)
    print('TrainNoisySet:',TrainNoisySet.shape)
    print('ValCleanSet:',ValCleanSet.shape)
    print('ValNoisySet:',ValNoisySet.shape)
    print('TestCleanSet:',TestCleanSet.shape)
    print('TestNoisytSet:',TestNoisytSet.shape)

    return TrainCleanSet,TrainNoisySet,ValCleanSet,ValNoisySet,TestCleanSet,TestNoisytSet

    '''
    noisy_ppg_path = '/home/wcj/CurrentProject/ECGDenoising/ppg_ma_0dB.mat';
    original_data = loadmat(noisy_ppg_path);
    noisy_ppg = original_data['sig']

    noisy_ppg_path = '/home/wcj/CurrentProject/ECGDenoising/ppg_bw_0dB.mat';
    original_data = loadmat(noisy_ppg_path);
    noisy_ppg = original_data['sig']
    print('noisy ppg shape', noisy_ppg.shape)

    noisy_ppg_path = '/home/wcj/CurrentProject/ECGDenoising/ppg_em_0dB.mat';
    original_data = loadmat(noisy_ppg_path);
    noisy_ppg = original_data['sig']
    print('noisy ppg shape', noisy_ppg.shape)
    '''
def train():
    pass

# init()
# save_slicesignal()
GetDataSet()
