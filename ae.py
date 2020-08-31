import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, nc, naf, latent, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        enc = nn.Sequential()
        enc.add_module('conv_layer_{0}_{1}'.format(nc, naf),
                       nn.Conv1d(nc, naf, 4, 2, 0,padding_mode='reflect',bias=True))
        # enc.add_module('MaxPool1d',nn.MaxPool1d(3,2))
        enc.add_module('batch_norm_{0}'.format(naf), nn.BatchNorm1d(naf))
        # enc.add_module('leaky_relu_{0}'.format(naf), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('leaky_relu_{0}'.format(naf), nn.PReLU())
        enc.add_module('conv_layer_{0}_{1}'.format(naf, naf * 2),
                       nn.Conv1d(naf, naf * 2, 4, 2, 0, padding_mode='reflect',bias=True))
        # enc.add_module('MaxPool1d', nn.MaxPool1d(3, 2))

        enc.add_module('batch_norm_{0}'.format(naf * 2), nn.BatchNorm1d(naf * 2))
        # enc.add_module('leaky_relu_{0}'.format(naf * 2), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('leaky_relu_{0}'.format(naf*2), nn.PReLU())
        enc.add_module('conv_layer_{0}_{1}'.format(naf * 2, naf * 4),
                       nn.Conv1d(naf * 2, naf * 4, 4, 2, 0,padding_mode='reflect', bias=True))
        # enc.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        enc.add_module('batch_norm_{0}'.format(naf * 4), nn.BatchNorm1d(naf * 4))
        # enc.add_module('leaky_relu_{0}'.format(naf * 4), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('leaky_relu_{0}'.format(naf * 4), nn.PReLU())
        enc.add_module('conv_layer_{0}_{1}'.format(naf * 4, naf * 8),
                       nn.Conv1d(naf * 4, naf * 8, 4, 2, 0, padding_mode='reflect', bias=True))
        # enc.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        enc.add_module('batch_norm_{0}'.format(naf * 8), nn.BatchNorm1d(naf * 8))
        # enc.add_module('leaky_relu_{0}'.format(naf * 4), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('leaky_relu_{0}'.format(naf * 8), nn.PReLU())
        enc.add_module('conv_layer_{0}_{1}'.format(naf * 8, latent),
                       nn.Conv1d(naf * 8, latent, 4, 2, 0,padding_mode='reflect', bias=True))
        # enc.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        enc.add_module('batch_norm_{0}'.format(latent), nn.BatchNorm1d(latent))
        # enc.add_module('leaky_relu_{0}'.format(latent), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('leaky_relu_{0}'.format(latent), nn.PReLU())

        self.enc = enc

    def forward(self, input):
        #print('Encoder input shape:', input.shape)
        input=input.unsqueeze(1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.enc, input, range(self.ngpu))
        else:
            output = self.enc(input)
            # print('Encoder output shape:', output.shape)
        return output


class Decoder(nn.Module):
    def __init__(self, nc, naf, latent, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        dec = nn.Sequential()
        dec.add_module('convt_layer_{0}_{1}'.format(latent, naf * 8),
                       nn.ConvTranspose1d(latent, naf * 8, 4, 2, 0,output_padding=0,bias=True))
        # dec.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        dec.add_module('batch_norm_{0}'.format(naf * 8), nn.BatchNorm1d(naf * 8))
        # dec.add_module('leaky_relu_{0}'.format(naf * 4), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('leaky_relu_{0}'.format(naf * 8), nn.PReLU())

        dec.add_module('convt_layer_{0}_{1}'.format(naf * 8, naf * 4),
                       nn.ConvTranspose1d(naf * 8, naf * 4, 5, 2, 0, output_padding=0, bias=True))
        # dec.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        dec.add_module('batch_norm_{0}'.format(naf * 4), nn.BatchNorm1d(naf * 4))
        # dec.add_module('leaky_relu_{0}'.format(naf * 2), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('leaky_relu_{0}'.format(naf * 2), nn.PReLU())

        dec.add_module('convt_layer_{0}_{1}'.format(naf * 4, naf * 2),
                       nn.ConvTranspose1d(naf * 4, naf * 2, 4, 2, 0,output_padding=0, bias=True))
        # dec.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        dec.add_module('batch_norm_{0}'.format(naf * 2), nn.BatchNorm1d(naf * 2))
        # dec.add_module('leaky_relu_{0}'.format(naf * 2), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('leaky_relu_{0}'.format(naf * 2), nn.PReLU())

        dec.add_module('convt_layer_{0}_{1}'.format(naf * 2, naf),
                       nn.ConvTranspose1d(naf * 2, naf, 4, 2, 0, output_padding=0,bias=True))
        # dec.add_module('MaxPool1d', nn.MaxPool1d(3, 2))
        dec.add_module('batch_norm_{0}'.format(naf), nn.BatchNorm1d(naf))
        # dec.add_module('leaky_relu_{0}'.format(naf), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('leaky_relu_{0}'.format(naf), nn.PReLU())

        dec.add_module('convt_layer_{0}_{1}'.format(naf, nc),
                       nn.ConvTranspose1d(naf, nc, 6, 2, 0,output_padding=0,bias=True))

        self.dec = dec

    def forward(self, input):
        #print('Decoder input shape:', input.shape)
        # input = input.unsqueeze(1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.dec, input, range(self.ngpu))
        else:
            output = self.dec(input)
            #print('Decoder output shape:',output.shape)

        return output


class AutoEncoder(nn.Module):
    def __init__(self, nc, naf, latent, ngpu):
        super(AutoEncoder, self).__init__()
        self.ngpu = ngpu

        ae = nn.Sequential()
        ae.add_module('Encoder', Encoder(nc, naf, latent, ngpu))
        ae.add_module('Decoder', Decoder(nc, naf, latent, ngpu))

        self.ae = ae

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def forward(self, input):
        # print('input shape:',input.shape)
        # normailze in the internal model
        # ori=input
        # mean=torch.mean(input,dim=1, keepdim=True)
        # std=torch.std(input,dim=1,unbiased=False, keepdim=True)
        # # print(mean.shape,std.shape)
        # input-=mean
        # input/=(std+1e-7)
        # for i in range(input.shape[0]):
        #     plt.plot(ori[i].detach().cpu().numpy(),label='original')
        #     plt.plot(input[i].detach().cpu().numpy(),label='scale')
        #     plt.legend()
        #     plt.show()
        #     if i==1:
        #         break
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.ae, input, range(self.ngpu))
        else:
            output = self.ae(input)
        output=output.squeeze(1)
        return output

class LinearAutoEncoder(nn.Module):
    # 输入通道，基础通道数，编码器输出通道，GPU个数
    def __init__(self, nc, naf, latent, ngpu):
        super(LinearAutoEncoder, self).__init__()
        self.encodersequentials=nn.Sequential(nn.Linear(nc,naf*2),
                                              nn.PReLU(),
                                              nn.Linear(naf * 2, naf),
                                              nn.PReLU(),
                                              nn.Linear(naf,int(naf*0.5)),
                                              nn.PReLU(),
                                              nn.Linear(int(naf*0.5),int(naf*0.25)),
                                              nn.Linear(int(naf * 0.25), latent),
                                              nn.PReLU()
                                              )
        # self.encodersequentials.append(nn.Linear(nc,naf*2))
        # self.encodersequentials.append(nn.PReLU(naf*2))
        # self.encodersequentials.append(nn.Linear(naf*2,naf*4))
        # self.encodersequentials.append(nn.PReLU(naf*4))
        # # self.encodersequentials.append(nn.Linear(naf * 4, naf * 8))
        # # self.encodersequentials.append(nn.PReLU(naf * 8))
        # self.encodersequentials.append(nn.Linear(naf * 4, latent))
        # self.encodersequentials.append(nn.PReLU(latent))

        self.decodersequentials=nn.Sequential(nn.Linear(latent, int(naf * 0.25)),
                                              nn.PReLU(),
                                              nn.Linear(int(naf * 0.25), int(naf * 0.5)),
                                              nn.PReLU(),
                                              nn.Linear(int(naf * 0.5), naf),
                                              nn.PReLU(),
                                              nn.Linear(naf,naf*2),
                                              nn.PReLU(),
                                              nn.Linear(naf*2,nc),
                                              nn.PReLU()
                                              )
        # self.decodersequentials.append(nn.Linear(latent, naf * 4))
        # self.decodersequentials.append(nn.PReLU(naf * 4))
        # # self.decodersequentials.append(nn.Linear(naf * 8, naf * 4))
        # # self.decodersequentials.append(nn.PReLU(naf * 4))
        # self.decodersequentials.append(nn.Linear(naf * 4, naf * 2))
        # self.decodersequentials.append(nn.PReLU(naf * 2))
        # self.decodersequentials.append(nn.Linear(naf * 2, nc))
        # self.decodersequentials.append(nn.PReLU(nc))
    def forward(self,input):
        input=input.unsqueeze(1)
        x=self.encodersequentials(input)
        output=self.decodersequentials(x)
        output = output.squeeze(1)
        return  output

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

if __name__=='__main__':
    input=torch.rand(1,1000)
    DAE=AutoEncoder(1,32,512,1)
    print('DAE parameters numbers:',DAE.get_n_params())
    output=DAE(input)
    print('DAE output shape:',output.shape)

    LinearDAE=LinearAutoEncoder(1000,32,4,1)
    print('LinearDAE parameters numbers:', LinearDAE.get_n_params())
    output = LinearDAE(input)
    print('LinearDAE output shape:', output.shape)
