from  model import  *
from PreProcess import  *
from ae import  *
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
from torch.optim import lr_scheduler # 学习率衰减
import os
# TrainNoisySet=[]
# ValNoisySet=[]
# TestNoisytSet=[]
#
# TrainCleanSet=[]
# ValCleanSet=[]
# TestCleanSet=[]


def save_slicesignal():
    ppgsignal=[]
    syncleanfraq=[]
    synnoisyfraq={}
    noisyfrag=[]
    SignalLength=1000

    ppgpath1 = '/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg.mat'
    ppgpath2 = '/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg1.mat'
    # ecgpath='/home/wcj/CurrentProject/EmotionRecongntion/dreamer-data/data1.npy'
    ecgpath1='/home/wcj/CurrentProject/ECGDenoising/mit_sample_ecg.mat'
    ecgpath2='/home/wcj/CurrentProject/ECGDenoising/ecg_wcj_8_27_denoise.mat'
    ecgpath3='/home/wcj/CurrentProject/ECGDenoising/ecg_8_15_bandpass.mat'
    # mitecg='/ho'

    empath = '/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/em'
    bwpath = '/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/bw'
    mapath = '/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/ma'

    # ecgsignal=load_ecg(ecgpath1)
    ecgsignal=load_ecgsegment(ecgpath3)
    # ecgsignal=ecgsignal*10
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
    em_data=em_data.reshape(1,-1)
    bw_data=bw_data.reshape(1,-1)
    ma_data=ma_data.reshape(1,-1)

    Repetive_em_data=[]
    Repetive_bw_data = []
    Repetive_ma_data = []
    for i in range(5):
        Repetive_em_data.append(em_data)
        Repetive_ma_data.append(ma_data)
        Repetive_bw_data.append(bw_data)
    Repetive_em_data=np.array(Repetive_em_data)
    Repetive_bw_data = np.array(Repetive_bw_data)
    Repetive_ma_data = np.array(Repetive_ma_data)
    Repetive_em_data=Repetive_em_data.reshape(1,-1)
    Repetive_ma_data=Repetive_ma_data.reshape(1,-1)
    Repetive_bw_data=Repetive_bw_data.reshape(1,-1)
    # print(Repetive_em_data.shape)
    # return

    # print(ppgsignal1.shape, ppgsignal2.shape,ecgsignal.shape, bw_data.shape)
    print('noisy data shape:',Repetive_em_data.shape)
    print('ecgsinal shape:',ecgsignal.shape)
    ppgsignal = np.concatenate((ppgsignal1, ppgsignal2), axis=1)
    # ppgsignal.append(ppgsignal1)
    # ppgsignal.append(ppgsignal2)
    # ppgsignal=ppgsignal1+ppgsignal2
    # ppgsignal=np.array(ppgsignal)
    # print(ppgsignal.shape)
    print('ppgsignal shape:',ppgsignal.shape)

    def slice_signal(Clean, Noisy, length,k):
        CleanLength = Clean.shape[1]
        NoisyLength = Noisy.shape[1]
        # Clean=preprocessing.scale(Clean)
        # print(Clean.shape)
        TotalLength = min(CleanLength, NoisyLength)

        '''
        snr
        '''
        # for j in range(0, 60, 5):
        # snr
        for j in range(0, 20, 5):
            # signal arrange
            for i in range(0, TotalLength - length, length):

                # print(i)
                # for i in range(1):
                fragcleansig = Clean[0, i:i +length].reshape(-1)
                # fragcleansig = preprocessing.scale(fragcleansig)
                fragnoisy = Noisy[0, i:i + length].reshape(-1)
                syncleanfraq.append(fragcleansig)
                noisyfrag.append(fragnoisy)
                '''
                calculate gain
                '''
                gain = calculategain(j, fragcleansig, fragnoisy)
                # print("gain:",gain)
                # gain=1
                key = str(j) + "dB"

                # if k==0 and i==0:
                if i==0:
                    synnoisyfraq[key] = [generate_signal(gain, fragcleansig, fragnoisy)]
                else:
                    synnoisyfraq[key].append(generate_signal(gain, fragcleansig, fragnoisy))
    '''
    save clean,noisy,synthetic signal
    '''

    TotalNoisyData = {}
    # TotalNoisyData['ma'] = ma_data
    # TotalNoisyData['bw'] = bw_data
    # TotalNoisyData['em'] = em_data
    TotalNoisyData['ma'] = Repetive_ma_data
    TotalNoisyData['bw'] = Repetive_bw_data
    TotalNoisyData['em'] = Repetive_em_data
    TypeNoisy = ['ma', 'bw', 'em']

    for i, type in enumerate(TypeNoisy):
        # syncleanfraq = []
        CurNoisylength=TotalNoisyData[type].shape[1]
        synnoisyfraq = {}
        noisyfrag = []
        # current noisy signal length < pure noisy signal length
        # if ecgsignal.shape[1]>CurNoisylength:
        #     for k in range(int(ecgsignal.shape[1]/CurNoisylength)):
        #         # print(k)
        #         TempEcg=ecgsignal[0,k*CurNoisylength:(k+1)*CurNoisylength]
        #         TempEcg=TempEcg.reshape(1,-1)
        #         slice_signal(TempEcg, TotalNoisyData[type], SignalLength,k)
        #     # print(np.array(syncleanfraq).shape)
        #         if k==0:
        #             noisyname = ("{}.mat".format(type))
        #             savemat(noisyname, {"sig": np.array(noisyfrag)})
        # else:

        slice_signal(ecgsignal,TotalNoisyData[type],SignalLength,0)
        for j in range(0, 20, 5):
            key = str(j) + "dB"
            # print(type, key)
            syntheticname = ("newOwnCollectData/ownecg_{}_{}".format(type, key))
            savemat(syntheticname, {"sig": np.array(synnoisyfraq[key])})
    cleanname = ("newOwnCollectData/ownecg_clean.mat")
    savemat(cleanname, {"sig": np.array(syncleanfraq)})





# slice_signal(ppgsignal,em_data,1000)
# print(np.array(syncleanfraq).shape)
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
#     tempsig=np.array(synnoisyfraq[key])
#     for k in range(tempsig.shape[0]):
#         # plt.plot(syncleanfraq[k],label='original')
#         plt.plot(syncleanfraq[k],label='scale')
#         plt.plot(tempsig[k],label='noisy')
#         plt.title("snr:{},noisy:em".format(key))
#         plt.legend()
#         plt.show()
#         break


def GetDataSet():
    total_noisy_sig=[]
    # clean_sig_path='/home/wcj/CurrentProject/ECGDenoising/ppg_clean.mat'
    # clean_sig_path='/home/wcj/CurrentProject/ECGDenoising/mitecg_clean.mat'
    clean_sig_path = '/home/wcj/CurrentProject/ECGDenoising/newOwnCollectData/ownecg_clean.mat'
    original_data=loadmat(clean_sig_path);
    clean_sig=original_data['sig']
    print('clean sig shape:',clean_sig.shape)

    TypeNoisy = ['ma', 'bw', 'em']
    for i, type in enumerate(TypeNoisy):
        for j in range(0, 20, 5):
            key = str(j) + "dB"
            # print(type, key)
            noisy_sig_path='/home/wcj/CurrentProject/ECGDenoising/newOwnCollectData/ownecg_'+str(type)+'_'+key+'.mat'
            original_data = loadmat(noisy_sig_path);
            noisy_sig = original_data['sig']
            total_noisy_sig.append(noisy_sig)
    total_noisy_sig=np.array(total_noisy_sig).reshape(-1,1000)
    print('noisy sig shape',total_noisy_sig.shape)



    '''
    split the noisy and respond clean data to make train data
    '''
    total_count=clean_sig.shape[0]
    # print(total_count)
    index1=int(total_count*(2/3))
    index2=int(total_count*(5/6))
    # print(index1,index2)
    TrainCleanSet=clean_sig[0:index1][:]
    TrainNoisySet=total_noisy_sig[0:index1][:]

    ValCleanSet=clean_sig[index1:index2][:]
    ValNoisySet=total_noisy_sig[index1:index2][:]

    TestCleanSet=clean_sig[index2:total_count][:]
    TestNoisytSet=total_noisy_sig[index2:total_count][:]

    print('TrainCleanSet:',TrainCleanSet.shape)
    print('TrainNoisySet:',TrainNoisySet.shape)
    print('ValCleanSet:',ValCleanSet.shape)
    print('ValNoisySet:',ValNoisySet.shape)
    print('TestCleanSet:',TestCleanSet.shape)
    print('TestNoisytSet:',TestNoisytSet.shape)

    return TrainCleanSet,TrainNoisySet,ValCleanSet,ValNoisySet,TestCleanSet,TestNoisytSet

    '''
    noisy_sig_path = '/home/wcj/CurrentProject/ECGDenoising/ppg_ma_0dB.mat';
    original_data = loadmat(noisy_sig_path);
    noisy_sig = original_data['sig']

    noisy_sig_path = '/home/wcj/CurrentProject/ECGDenoising/ppg_bw_0dB.mat';
    original_data = loadmat(noisy_sig_path);
    noisy_sig = original_data['sig']
    print('noisy ppg shape', noisy_sig.shape)

    noisy_sig_path = '/home/wcj/CurrentProject/ECGDenoising/ppg_em_0dB.mat';
    original_data = loadmat(noisy_sig_path);
    noisy_sig = original_data['sig']
    print('noisy ppg shape', noisy_sig.shape)
    '''
class OriData(data.Dataset):
    def __init__(self, clean,noisy):
        self.cleandata = clean
        self.noisydata = noisy
    def __getitem__(self, index):
        cleansignal = self.cleandata[index]
        noisysignal=self.noisydata[index]
        return cleansignal,noisysignal
    def __len__(self):
        return self.cleandata.shape[0]
def trainAutoEncoder():
    batch_size=128
    epochs=1
    iteration = 0
    numepoch = 0
    device = 'cpu'
    if torch.cuda.is_available:
        # device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        device = 'cuda'
    CUDA = (device == 'cuda')
    TrainCleanSet, TrainNoisySet, ValCleanSet, ValNoisySet, TestCleanSet, TestNoisytSet = GetDataSet()
    DAE = AutoEncoder(1, 32, 512, 1)
    print('Total model parameters: ', DAE.get_n_params())
    trainset=OriData(TrainCleanSet,TrainNoisySet)
    valset=OriData(ValCleanSet,ValNoisySet)
    testset=OriData(TestCleanSet,TestNoisytSet)
    print(len(trainset),len(valset),len(testset))
    TrainLoader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=CUDA)
    ValLoader=DataLoader(valset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=CUDA)
    TestLoader=DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=CUDA)
    print(TrainLoader.__len__(),ValLoader.__len__(),TestLoader.__len__())
    DAE.to(device)
    # DAE=DAE.double()
    optimizer = optim.Adam(DAE.parameters(), lr=0.001)
    # optimizer = optim.RMSprop(DAE.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, 10, 0.1)  # # 每过3个epoch，学习率乘以0.1
    criterion = nn.MSELoss().to(device)
    # criterion=nn.L1Loss().to(device)
    # criterion=nn.BCELoss().to(device)
    for epoch in range(epochs):
        numepoch += 1
        for i, data in enumerate(TrainLoader,start=1):
            iteration += 1
            error=0.0
            if len(data)==2:
                clean,noisy=data
            else:
                raise ValueError('Returned {} elements per '
                                 'sample?'.format(len(data)))
            # for k in range(5):
            #     plt.plot(clean[k],label='clean')
            #     plt.plot(noisy[k],label='noisy')
            #     plt.legend()
            #     plt.show()
            # return
            # clean = clean.unsqueeze(1)
            # noisy = noisy.unsqueeze(1)
            # print(clean)
            clean = clean.to(device)
            noisy = noisy.to(device)
            noisy=noisy.float()
            clean=clean.float()
            # print(noisy.shape)
            optimizer.zero_grad()
            output=DAE(noisy)
            loss=criterion(output,clean)
            # print(output.shape,clean.shape)
            loss.backward()
            optimizer.step()
            error+=loss.item()
            if i % 10 == 9:
                print('[%d,%5d] loss %f' % (epoch + 1, i + 1, error/10))
                error = 0.0

    print('finish training')
    savemodelname=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+('epoch-%d-iteration%d.pt'%(numepoch,iteration))
    model = DAE.cpu()
    model=model.eval()
    x = torch.rand(1, 1000)
    # x=x.double()
    traced_module = torch.jit.trace(model, x)
    traced_module.save(savemodelname)
    # torch.save(DAE.state_dict(),savemodelname)

def train():
    model=ECGDAE(1, 1000).build()
    model=ECGDAE(1, 1000).build()
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss="mse", optimizer=opt,
                  metrics=[keras.metrics.MeanSquaredError()])



    import datetime

    logdir = "./keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    history = model.fit(x=TrainNoisySet, y=TrainCleanSet, batch_size=100,
        validation_data=(ValNoisySet, ValCleanSet),
        epochs=100, verbose=1, callbacks=[tensorboard_callback])

    model.save('./data/tf_model_savedmodel', save_format="tf")
    print('export saved model.')
# steps_per_epoch=len(TrainCleanSet) // 100,
# init()
def TestAutoEncoder():

    '''
    20200829-112830epoch-50-iteration11000  train on self-collected signal
    20200829-114619epoch-100-iteration22000.pt train on self-collected signal (only wcj)
    20200829-172054epoch-50-iteration16250.pt train on self-collected signal (five people data)
    '''

    # model=AutoEncoder(1, 32, 256, 1)
    device = 'cpu'
    if torch.cuda.is_available:
        # device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        device = 'cuda'

    # model.load_state_dict(torch.load('./20200828-092545epoch-1-iteration325.pt'))
    model=torch.jit.load('./20200831-172652epoch-1-iteration382.pt')
    # model_dict = model.state_dict()
    # print(model)
    TrainCleanSet, TrainNoisySet, ValCleanSet, ValNoisySet, TestCleanSet, TestNoisytSet = GetDataSet()
    model.to(device)
    # model=model.double()
    model.eval()
    for i in range(1):
    # for i in range(5000,5100,5):
    # for i in range(ValNoisySet.shape[0]):

        testdata=ValNoisySet[i]
        plt.plot(testdata,label='noisy')
        # testdata = preprocessing.scale(testdata)
        cleandata=ValCleanSet[i]
        plt.plot(cleandata,label='clean')
        testdata=testdata.reshape(1,-1)
        testdata=torch.from_numpy(testdata)
        testdata=testdata.to(device)
        testdata = testdata.float()
        print('******:',testdata.type())
        # testdata = testdata.float()
        # print(testdata.shape)
        output=model(testdata)
        print('******:',output.type())
        # print(output.shape)
        output=output.detach().cpu().numpy()
        plt.plot(output[0],label='denoise')
        plt.legend()
        plt.show()

        # break
def TestOnRealNoisyData():
    device = 'cpu'
    if torch.cuda.is_available:
        # device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        device = 'cuda'

    # model.load_state_dict(torch.load('./20200828-092545epoch-1-iteration325.pt'))
    model = torch.jit.load('./20200831-143649epoch-50-iteration19100.pt')
    # model_dict = model.state_dict()
    # print(model)
    noisysig=load_ecgsegment('./ecg_wcj_8_27_noisy.mat')
    print(noisysig.shape)
    model.to(device)
    model = model.double()
    model.eval()
    for i in [4, 5]:
        # for i in range(ValNoisySet.shape[0]):

        testdata = noisysig[i]
        testdata=preprocessing.scale(testdata)
        testdata = testdata.reshape(1, -1)
        plt.plot(testdata[0],label='noisy')
        testdata = torch.from_numpy(testdata)
        testdata = testdata.to(device)

        # print(testdata.shape)
        output = model(testdata)
        output = output.detach().cpu().numpy()
        print(output.shape)
        plt.plot(output[0], label='denoise')
        plt.legend()
        plt.show()

def test():
    model=keras.models.load_model('./data/tf_model_savedmodel')
    GeneratePPG=model.predict(TestNoisytSet)
    TotalAE=[]
    for i in range(TestNoisytSet.shape[0]):
        cur_snr = snr_pred(TestCleanSet[i], GeneratePPG[i])
        ori_snr = snr_pred(TestCleanSet[i], TestNoisytSet[i])
        ae=abs(cur_snr-ori_snr)
        TotalAE.append(ae)
    print('average increased snr:',np.mean(TotalAE))
    '''
    visual
    '''
    fig = plt.figure(figsize=(20, 100))
    for i in range(10):
        fig.add_subplot(10, 1, i + 1)
        # if i % 2 == 0:
        plt.plot(GeneratePPG[int(i/2)],label='denoise')
        plt.plot(TestCleanSet[int(i/2)],label='clean')
        plt.plot(TestNoisytSet[int(i/2)],label='noisy')
        plt.legend()
        cur_snr=snr_pred(TestCleanSet[int(i/2)],GeneratePPG[int(i/2)])
        ori_snr=snr_pred(TestCleanSet[int(i/2)],TestNoisytSet[int(i/2)])
        plt.title('ori_snr:'+str(ori_snr)+' cur_snr:'+str(cur_snr))
        # else:
        #     plt.plot(TestCleanSet[int(i/2)],label='clean')
        #     plt.title('clean')
    plt.show()
    # print(GeneratePPG.shape)

if __name__=='__main__':
    # load_ecg('/home/wcj/CurrentProject/EmotionRecongntion/dreamer-data/data1.npy')
    # save_slicesignal()

    #show clean and noisy data
    # TrainCleanSet, TrainNoisySet, ValCleanSet, ValNoisySet, TestCleanSet, TestNoisytSet=GetDataSet()
    # TrainCleanSet = TrainCleanSet.reshape(-1,1000,1)
    # TrainNoisySet = TrainNoisySet.reshape(-1, 1000,1)
    # ValCleanSet = ValCleanSet.reshape(-1, 1000,1)
    # ValNoisySet = ValNoisySet.reshape(-1, 1000,1)
    # TestCleanSet = TestCleanSet.reshape(-1, 1000,1)
    # TestNoisytSet = TestNoisytSet.reshape(-1, 1000,1)
    # for i in range(5000,5100,5):
    #     plt.plot(ValCleanSet[i],label='clean')
    #     plt.plot(ValNoisySet[i],label='noisy')
    #     plt.legend()
    #     plt.show()


    # print(TrainCleanSet.shape,TrainNoisySet.shape)
    # train()
    # test()
    # trainAutoEncoder()
    TestAutoEncoder()
    # TestOnRealNoisyData()


    # normalize in pytorch demo
    # a=np.array([[1.,2.,3.],[4.,5.,6.]])
    # a=torch.from_numpy(a)
    # input=a
    # mean = torch.mean(input, dim=1, keepdim=True)
    # std = torch.std(input, dim=1, unbiased=False, keepdim=True)
    # print(mean)
    # print(std)
    # input-=mean
    # input/=std
    # print(input)

    # apped array reshape demo
    # a=np.array([[1,2,3,4]])
    # b=np.array([[5,6,7,8]])
    # c=np.array([[9,10,11,12]])
    # d=[]
    # d.append(a)
    # d.append(b)
    # d.append(c)
    # d=np.array(d)
    # d=d.reshape(1,-1)
    # print(d)

    # x=torch.rand(1,1000)
    # print(x.shape)
    # # x=torch.from_numpy(x)
    # x=x.unsqueeze(1)
    # # x=x.unsqueeze(2)
    # print(x.shape)