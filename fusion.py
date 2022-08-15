from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.nn import Parameter
from torch.nn import functional as F
import numpy as np

import torch.backends.cudnn as cudnn
from options import parse_args
opt = parse_args()
class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims,rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims
        self.video_in = input_dims
        self.text_in = input_dims

        self.output_dim = input_dims
        self.rank = rank
        self.use_softmax = use_softmax

        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_in + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_in + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_in + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal(self.audio_factor)
        xavier_normal(self.video_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = audio_x
        video_h = video_x
        text_h = text_x
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), audio_h), dim=1)
        _video_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), video_h), dim=1)
        _text_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output
class CatFusion(nn.Module):
    def __init__(self,in_size, dropout=0.1):
        super(CatFusion, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size*3, 512), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder=nn.Sequential(encoder1,encoder2,encoder3,encoder4)
    def forward(self,x_gene,x_path,x_cna):
        cat=torch.cat((x_gene,x_path,x_cna),dim=1)
        out=self.encoder(cat)
        return out

class CatFusion2(nn.Module):
    def __init__(self,in_size, dropout=0.1):
        super(CatFusion2, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size*2, 512), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(256, 80), nn.ReLU(), nn.Dropout(p=dropout))
        #encoder4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder=nn.Sequential(encoder1,encoder2,encoder3)
    def forward(self,x1,x2):
        cat=torch.cat((x1,x2),dim=1)
        out=self.encoder(cat)
        return out

class Atten(nn.Module):
    def __init__(self, in_dim):
        super(Atten, self).__init__()
        self.chanel_in = in_dim
        self.query_linear = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.key_linear = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.value_linear = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.gamma = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.softmax = nn.Softmax(dim=1)  #

    def forward(self, x1,x2):
        """
            inputs :
                x : input feature maps(B*features)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, features = x1.size()
        proj_query = self.query_linear(x1).view(m_batchsize, -1, features).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_linear(x2).view(m_batchsize, -1, features)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_linear(x2).view(m_batchsize, -1, features)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, features)
        return out* self.gamma
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_linear =  nn.Sequential(nn.Linear(in_dim,in_dim),nn.ReLU())
        self.key_linear =  nn.Sequential(nn.Linear(in_dim,in_dim),nn.ReLU())
        self.value_linear =  nn.Sequential(nn.Linear(in_dim,in_dim),nn.ReLU())
        self.gamma = Parameter(torch.FloatTensor([1]), requires_grad=False)
        self.softmax = nn.Softmax(dim=0)  #
    def forward(self, x):
        """
            inputs :
                x : input feature maps(B*features)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, features = x.size()
        proj_query = self.query_linear(x).view(m_batchsize, -1,features).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_linear(x).view(m_batchsize, -1, features)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value=self.value_linear(x)
        proj_value = x.view(m_batchsize, -1, features)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, features)

        out = self.gamma * out + x
        return out


class AttentionDot(nn.Module):
    def __init__(self):
        super(AttentionDot,self).__init__()
        self.softmax=nn.Softmax()
    def forward(self,k,q,v):
        w=self.softmax(self.softmax(torch.matmul(k,q.permute(0,2,1))))
        out=torch.matmul(w,v)
        return out
class Self_Attn1(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn1, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize,width, height)

        out = self.gamma * out + x
        return out, attention
class mfb(nn.Module):
    def __init__(self, in_size,MFB_FACTOR_NUM=1,dropout=0.1):
        super(mfb, self).__init__()
        ############
        self.MFB_FACTOR_NUM=MFB_FACTOR_NUM
        self.in_size=in_size
        self.Linear_dataproj1 = nn.Linear(in_size,in_size*MFB_FACTOR_NUM)
        self.Linear_imgproj1 = nn.Linear(in_size,in_size*MFB_FACTOR_NUM)
        self.Linear_predict1 = nn.Linear(in_size,in_size)
        self.dropout=dropout
        #####################
        #self.Linear_dataproj2 = nn.Linear(in_size, in_size * MFB_FACTOR_NUM)
        #self.Linear_imgproj2 = nn.Linear(in_size, in_size * MFB_FACTOR_NUM)
        #self.Linear_predict2 = nn.Linear(in_size, in_size * MFB_FACTOR_NUM)

        #####################
        #self.Linear_dataproj3 = nn.Linear(in_size, in_size * MFB_FACTOR_NUM)
        #self.Linear_imgproj3 = nn.Linear(in_size, in_size * MFB_FACTOR_NUM)
        #self.Linear_predict3 = nn.Linear(in_size, in_size * MFB_FACTOR_NUM)
    def forward(self,x_gene,x_patho):
        x_gene_out1= self.Linear_dataproj1(x_gene)  # data_out (batch, in_size)
        x_patho_out1 = self.Linear_imgproj1(x_patho)  # img_feature (batch, in_size)
        iq1 = torch.mul(x_gene_out1, x_patho_out1)
        iq1 = F.dropout(iq1,self.dropout)
        iq1 = iq1.view(-1, 1, self.in_size,self.MFB_FACTOR_NUM)
        iq1 = torch.squeeze(torch.sum(iq1, 3))  # sum pool

        iq1 = torch.sqrt(F.relu(iq1)) - torch.sqrt(F.relu(-iq1))  # signed sqrt
        iq1 = F.normalize(iq1)
        iq1 = self.Linear_predict1(iq1)  # (64,3000)
        iq1 = F.log_softmax(iq1)

        #################
        #x_gene_out2 = self.Linear_dataproj2(x_gene)  # data_out (batch, in_size)
        #x_cna_out2 = self.Linear_imgproj2(x_cna)  # img_feature (batch, in_size)
        #iq2 = torch.mul(x_gene_out2, x_cna_out2)
        #iq2 = F.dropout(iq2, self.dropout, training=self.training)
        #iq2 = iq2.view(-1, 1, 32, self.MFB_FACTOR_NUM)
        #iq2 = torch.squeeze(torch.sum(iq2, 3))  # sum pool
        #iq2 = torch.sqrt(F.relu(iq2)) - torch.sqrt(F.relu(-iq2))  # signed sqrt
        #iq2 = F.normalize(iq2)
        #iq2 = self.Linear_predict1(iq2)  # (64,3000)
        #iq2 = F.log_softmax(iq2)

        #################
        #x_patho_out3 = self.Linear_dataproj3(x_patho)  # data_out (batch, in_size)
        #x_cna_out3 = self.Linear_imgproj3(x_cna)  # img_feature (batch, in_size)
        #iq3 = torch.mul(x_patho_out3, x_cna_out3)
        #iq3 = F.dropout(iq3, self.dropout, training=self.training)
        #iq3 = iq3.view(-1, 1, 32, self.MFB_FACTOR_NUM)
        #iq3 = torch.squeeze(torch.sum(iq3, 3))  # sum pool
        #iq3 = torch.sqrt(F.relu(iq3)) - torch.sqrt(F.relu(-iq3))  # signed sqrt
        #iq3 = F.normalize(iq3)
        #iq3 = self.Linear_predict1(iq3)  # (64,3000)
        #iq3 = F.log_softmax(iq3)

        return iq1

class GateFilter(nn.Module):
    def __init__(self,in_size):
        super(GateFilter,self).__init__()
        self.linear1=nn.Linear(in_size,in_size)
        self.linear2=nn.Linear(in_size,in_size)

        self.tanh=nn.Tanh()
        self.linear3 = nn.Linear(in_size, in_size)
        self.linear4 = nn.Linear(in_size, in_size)

        self.linear5 = nn.Linear(in_size, in_size)
        self.linear6 = nn.Linear(in_size, in_size)

        self.linear7 = nn.Sequential(nn.Linear(in_size,in_size), nn.ReLU())
        self.linear8 = nn.Sequential(nn.Linear(in_size,in_size), nn.ReLU())
        self.linear9 = nn.Sequential(nn.Linear(in_size,in_size), nn.ReLU())
 
    def forward(self,x1,x2,x3):
        x2_1=self.linear1(x2)
        x3_1=self.linear2(x3)
        x1_filter=self.tanh(x2_1+x3_1)
        x1_out=self.linear7(x1*x1_filter)

        x1_2 = self.linear3(x1)
        x3_2 = self.linear4(x3)
        x2_filter = self.tanh(x1_2+x3_2)
        x2_out =self.linear8(x2_filter*x2)

        x1_3 = self.linear5(x1)
        x2_3 = self.linear6(x2)
        x3_filter = self.tanh(x1_3+x2_3)
        x3_out =self.linear9(x3_filter*x3)

        return x1_out,x2_out,x3_out

class MLPAttention(nn.Module):
    def __init__(self,in_size):
        super(MLPAttention, self).__init__()
        self.linear1=nn.Sequential(nn.Linear(in_size,in_size),nn.Tanh())
        self.linear2 = nn.Sequential(nn.Linear(in_size,in_size),nn.Tanh())
        self.linear3 = nn.Sequential(nn.Linear(in_size,in_size),nn.Tanh())
        self.softmax=nn.Softmax(dim=1)
        self.linear=nn.Sequential(nn.Linear(in_size,in_size),nn.ReLU())
    def forward(self,x1,x2,x3):
        w1=self.linear1(x1)
        w1=self.softmax(w1)
        w2 = self.linear1(x2)
        w2 = self.softmax(w2)
        w3 = self.linear1(x3)
        w3 = self.softmax(w3)
        out=w1*x1+w2*x2+w3*x3
        out=self.linear(out)
        return out

class MCF(nn.Module):
    def __init__(self,in_size):
        super(MCF,self).__init__()
        self.in_size=in_size
    def forward(self,x1,x2,x3):
        x_g=shift(x1)
        x_p=shift(x2)
        x_c=shift(x3)
        x1=x1.view(-1,1,self.in_size)
        x2 = x2.view(-1, 1, self.in_size)
        x3 = x3.view(-1, 1, self.in_size)
        x_g=torch.bmm(x1,x_g).view(x1.size(0),-1)
        x_p = torch.bmm(x2,x_p).view(x1.size(0),-1)

        x_c= torch.bmm(x3,x_c).view(x1.size(0),-1)

        return x_g,x_p,x_c
def shift(x):
    cudnn.deterministic = True
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    r = torch.ones((x.size(0), x.size(1), x.size(1)), dtype=torch.float32).to(device)
    for i in range(x.size(0)):
        temp = torch.ones((x.shape[1], x.shape[1]), dtype=torch.float32).to(device)
        temp[0] = x[i, :]
        for j in range(1, x.size(1)):
            # print(x[i,x.size[1]-j:])
            temp[j,:j] = x[i, x.size(1) - j:]
            temp[j,j:]= x[i,:x.size(1) - j]
        r[i]=temp
    return r

class Core_Fusion(nn.Module):

    def __init__(self, in_size,fac):
        super(Core_Fusion, self).__init__()
        # visul & text embeddings
        self.r=fac
        self.linear_v = nn.Linear(in_size,in_size)
        self.linear_t = nn.Linear(in_size,in_size)
        # Core tensor
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(in_size,in_size)
            for i in range(self.r)])

        self.list_linear_ht = nn.ModuleList([
            nn.Linear(in_size,in_size)
            for i in range(self.r)])

    def forward(self, input_v, input_t):
        batch_size_v = input_v.size(0)
        batch_size_t = input_t.size(0)

        x_v = self.linear_v(input_v)

        x_t = self.linear_t(input_t)
        x_mm = []

        for i in range(self.r):
            x_hv = F.dropout(x_v, p=0.1, training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)

            x_ht = F.dropout(x_t, p=0.1, training=self.training)
            x_ht = self.list_linear_ht[i](x_ht)
            x_mm.append(x_hv*x_ht)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1)
        return x_mm
class MLPAttention6(nn.Module):
    def __init__(self,in_size):
        super(MLPAttention6,self).__init__()
        self.linear1=nn.Sequential(nn.Linear(in_size,in_size),nn.Tanh())
        self.linear2 = nn.Sequential(nn.Linear(in_size, in_size), nn.Tanh())
        self.linear3 = nn.Sequential(nn.Linear(in_size, in_size), nn.Tanh())
        self.linear4 = nn.Sequential(nn.Linear(in_size, in_size), nn.Tanh())
        self.linear5 = nn.Sequential(nn.Linear(in_size, in_size), nn.Tanh())
        self.linear6 = nn.Sequential(nn.Linear(in_size, in_size), nn.Tanh())
        self.softmax=nn.Softmax()
        self.linear=nn.Linear(in_size,in_size)
    def forward(self,x1,x2,x3,x4,x5,x6):
        w1=self.linear1(x1)
        w1=self.softmax(w1)
        w2 = self.linear2(x2)
        w2 = self.softmax(w2)
        w3 = self.linear3(x3)
        w3 = self.softmax(w3)
        ##
        w4 = self.linear4(x4)
        w4 = self.softmax(w4)
        w5 = self.linear5(x5)
        w5 = self.softmax(w5)
        w6 = self.linear6(x6)
        w6 = self.softmax(w6)
        out=w1*x1+w2*x2+w3*x3+w4*x4+w5*x5+w6*x6
        out=self.linear(out)
        return out
class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, z):
        validity = self.model(z)
        return validity
