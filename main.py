from  model import  *
from PreProcess import  *

def slice_signal(Clean,Noisy,length):
    pass

TrainSet=[]
ValSet=[]
TestSet=[]
ppgsignal=[]

ppgpath1='/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg.mat'
ppgpath2='/home/wcj/CurrentProject/ppgalgorithm/Troika/Data/bandpassppg1.mat'
empath='/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/em'
bwpath='/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/bw'
mapath='/home/wcj/DataSet/physionet.org/files/nstdb/1.0.0/ma'

ppgsignal1=load_ppg(ppgpath1)
ppgsignal2=load_ppg(ppgpath2)
# ppgsignal1=np.reshape(ppgsignal1,(-1))
# ppgsignal2=np.reshape(ppgsignal2,(-1))
em_data=load_noisy(empath)
bw_data=load_noisy(bwpath)
ma_data=load_noisy(mapath)
print(ppgsignal1.shape,ppgsignal2.shape,em_data.shape)
ppgsignal=np.concatenate((ppgsignal1,ppgsignal2),axis=1)
# ppgsignal.append(ppgsignal1)
# ppgsignal.append(ppgsignal2)
# ppgsignal=ppgsignal1+ppgsignal2
# ppgsignal=np.array(ppgsignal)
# print(ppgsignal.shape)
print(ppgsignal.shape)


for i in range(5):
    pass