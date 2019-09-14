import torch
import torch.nn as nn
import numpy as np
import kgdlg.utils.print_utils as print_utils
import kgdlg.utils.data_utils as data_utils
import kgdlg.utils.time_utils as time_utils
import kgdlg.utils.operation_utils as operation_utils
from torch.autograd import Variable
import sys
import random

import torch.nn.functional as F
import kgdlg.gaussianMixtureModel as gaussianMixtureModel

class NMTModel(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src_inputs, tgt_inputs, src_lengths):
        

        # Run wrods through encoder

        enc_outputs, enc_hidden = self.encode(src_inputs, src_lengths, None)




        dec_init_hidden = self.init_decoder_state(enc_hidden, enc_outputs)
            
        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs, enc_outputs, dec_init_hidden
            )        

        return dec_outputs, attn


    def order_sequence(self, inputs, lengths):
        idx_len_pair = []
        for i,l in enumerate(lengths):
            idx_len_pair.append((i,l))
        sorted_idx_len_pair=sorted(idx_len_pair,key=lambda x:x[1],reverse=True)
        sorted_idx = []
        sorted_length = []
        for x in sorted_idx_len_pair:
            sorted_idx.append(x[0])
            sorted_length.append(x[1])

        order_inputs = inputs[:,sorted_idx]
        return order_inputs, sorted_length, sorted_idx
    def reorder_enc_seqence(self, inputs, sorted_idx):
        raw_sorted_pair = []
        for raw_idx,sorted_idx in enumerate(sorted_idx):
            raw_sorted_pair.append((raw_idx,sorted_idx))
        sorted_pair=sorted(raw_sorted_pair,key=lambda x:x[1],reverse=False)
        raw_indices = [x[0] for x in sorted_pair]
        reorder_inputs = inputs[:,raw_indices,:]
        return reorder_inputs


    def encode(self, input, lengths=None, hidden=None):
        order_inputs, sorted_length, sorted_idx =self.order_sequence(input,lengths)

        emb = self.enc_embedding(order_inputs)
        enc_outputs, enc_hidden = self.encoder(emb, sorted_length, None)
        enc_hidden = self.reorder_enc_seqence(enc_hidden,sorted_idx)
        enc_outputs = self.reorder_enc_seqence(enc_outputs,sorted_idx)
        return enc_outputs, enc_hidden

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input, context, state):
        emb = self.dec_embedding(input)
        dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, state
            )     

        return dec_outputs, dec_hiddens, attn
    
    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):   
        ckpt = torch.load(filename)
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder.load_state_dict(ckpt['encoder_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        epoch = ckpt['epoch']
        return epoch

class MoSGenerator(nn.Module):
    def __init__(self, n_experts, input_szie, output_size):
        super(MoSGenerator, self).__init__()
        self.input_szie = input_szie
        self.output_size = output_size
        self.n_experts = n_experts
        self.prior = nn.Linear(input_szie, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(input_szie, n_experts*input_szie), nn.Tanh())
        self.out_linear = nn.Linear(input_szie, output_size)
        self.softmax = nn.Softmax(-1)


    def forward(self, input):
        latent = self.latent(input)
        
        logits = self.out_linear(latent.view(-1, self.input_szie))
        # 960 ,5
        prior_logit = self.prior(input).contiguous().view(-1, self.n_experts)
        prior = self.softmax(prior_logit)
        prob = self.softmax(logits).view(-1, self.n_experts, self.output_size)

        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
        log_prob = torch.log(prob.add_(1e-8))

        return log_prob



# class RecognitionNetwork(nn.Module):

# class PriorNetwork(nn.Module):

class FCLayer(nn.Module):
    """ fully connected layer """
    def __init__(self, input_dim, output_dim, use_bias=True, multigpu=False) :
        self.multigpu = multigpu
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x):
        y = self.fc(x)
        return y

class VariationalInference(nn.Module):
     """ VariationalInference and GMM """
     def __init__(self, opt, fc_use_bias=True):
         #self.multigpu = opt.multigpu
         super(VariationalInference, self).__init__()
         
         encoder_dim = opt.hidden_size
         decoder_dim = opt.hidden_size
         self.opt = opt
         
         self.fc_z_mean = FCLayer(encoder_dim, opt.latent_size)         
         self.fc_z_log_variance_sq = FCLayer(encoder_dim, opt.latent_size)
         if self.opt.vae_type == 3:
             self.gmm = gaussianMixtureModel.gaussianMixtureModel(opt.latent_size, opt.cluster_num_gmm, opt.train_batch_size, opt)
             print("in init var out print gmm cluster_prior:", self.gmm.cluster_prior)
         self.use_gmm_output_fc = opt.use_gmm_output_fc
         if self.use_gmm_output_fc:
             self.fc_z_output = FCLayer(opt.latent_size, decoder_dim)         
         else:
             if decoder_dim != opt.latent_size:
                 print("Error! If there is no transforming(FC) layer after GMM, Must keep decoder_dim and latent_size equal. decoder_dim:", decoder_dim, "latent_size:", opt.latent_size)
                 assert decoder_dim == opt.latent_size

     def PostPrior_KL_in_VAE(self, mean, log_variance_sq):
         '''log_variance_sq is log((variance)**2)'''
         KL_buf_matrix = 1 + log_variance_sq - mean * mean - torch.exp(log_variance_sq)
         KL_in_batch = 0 - 0.5 * print_utils.sum_with_axis(KL_buf_matrix, [1])
         if self.opt.debug_mode >= 3:
             print("KL_buf_matrix in PostPrior_KL_in_VAE size:", KL_buf_matrix.size())
             print("KL_in_batch in PostPrior_KL_in_VAE size:", KL_in_batch.size())
         if self.opt.debug_mode >= 4:
             print("KL_buf_matrix in PostPrior_KL_in_VAE:", KL_buf_matrix)
             #print("KL_in_batch in PostPrior_KL_in_VAE:", KL_in_batch)
         return KL_in_batch

     def reparameter(self, mean, variance_sq):
         all_zero_mean = torch.zeros(mean.size()).cuda() # hard coding
         all_one_var_squ = torch.ones(variance_sq.size()).cuda()
         epsilon = torch.normal(all_zero_mean, all_one_var_squ)
         #a = variance_sq / 2
         #b = torch.exp(a)
         #print("variance_sq:", variance_sq.size(), "torch.exp(a):", b.size(), "epsilon:", epsilon.size(), "mean:", mean.size())
         #c = b * epsilon
         #d = mean + c
         #return d
         return mean + torch.exp(variance_sq / 2) * epsilon

     def forward(self, encoder_output, src_text=None, tgt_text=None):
         if self.opt.debug_mode >= 3:
             print("encoder_output size:", encoder_output.size())
         if self.opt.debug_mode >= 3:
             print("encoder_output:", encoder_output)
         encoder_output = encoder_output.view(encoder_output.size()[1:]) # eliminate first dimension (1*batch_size*dim) -> (batch_size*dim). To adapt to the shape of nn.RNN/GRU/LSTM output

         if self.opt.debug_mode >= 6:
             if self.opt.vae_type == 3:
                 print("0 out print gmm cluster_prior:", self.gmm.cluster_prior)
             print("encoder_output after reshape size:", encoder_output.size())
             print("encoder_output after reshape:", encoder_output)

         z_mean = self.fc_z_mean(encoder_output)
         if self.opt.debug_mode >= 3:
             print("z_mean after fc size:", z_mean.size())
         if self.opt.debug_mode >= 3:
             print("z_mean after fc:", z_mean)
         
         z_log_variance_sq = self.fc_z_log_variance_sq(encoder_output) # log((variance)**2)
         if self.opt.debug_mode >= 3:
             print("z_log_variance_sq after fc size:", z_log_variance_sq.size())
         if self.opt.debug_mode >= 3:
             print("z_log_variance_sq after fc:", z_log_variance_sq)
         
         z = self.reparameter(z_mean, z_log_variance_sq)
         if self.opt.debug_mode >= 3:
             print("z after reparameter size:", z.size())
         if self.opt.debug_mode >= 3:
             print("z after reparameter:", z)
         print_utils.save_latent_Z_with_text(src_text, z, None, None, opt)
         '''if self.opt.save_z_and_sample:
             #print_utils.print_matrix(z, sys.stderr)
             if not src_text is None:
                 #print("src_text is none z:", z)
                 src_text_t = torch.transpose(src_text, 0, 1)
                 print_utils.print_matrix_with_text(z, src_text_t, self.opt.variable_src_dict.itos, 4, sys.stderr)'''
         #return z_mean, z_log_variance_sq, z
         
         # repeat for category
         if self.opt.vae_type == 3:
             P_c_given_x, loss_without_crossent = self.gmm(z_mean, z_log_variance_sq, z) 
         elif self.opt.vae_type == 2 or self.opt.vae_type == 1: # VAE
             P_c_given_x = None
             loss_without_crossent = self.PostPrior_KL_in_VAE(z_mean, z_log_variance_sq)
         if self.opt.debug_mode >= 6:
             #print("after gmm before reshape size z:", z.size(), "P_c_given_x:", P_c_given_x.size(), "loss_without_crossent:", loss_without_crossent.size())
             print("after gmm before reshape size z:", z.size(), "loss_without_crossent:", loss_without_crossent.size())
         if self.opt.debug_mode >= 6:
             print("after gmm before reshape z:", z, "P_c_given_x:", P_c_given_x, "loss_without_crossent:", loss_without_crossent)
         
         if self.use_gmm_output_fc:
             z = self.fc_z_output(z)
         z = z.view([1] + [i for i in z.size()]) # add first dimension (batch_size*dim) -> (1*batch_size*dim). To adapt to the shape of nn.RNN/GRU/LSTM output
         if self.opt.debug_mode >= 3:
             print("variational output size z:", z.size(), "loss_without_crossent:", loss_without_crossent.size())
         if self.opt.debug_mode >= 3:
             print("variational output z:", z, "P_c_given_x:", P_c_given_x, "loss_without_crossent:", loss_without_crossent)
         
         return z, P_c_given_x, loss_without_crossent

class LatentNet(nn.Module):
    def __init__(self, opt):
        super(LatentNet, self).__init__()
        self.opt = opt
        
        if opt.vae_type in [8]:
            # Recognition Network
            if opt.variance_memory_type in [0, 2, 4]:
                recog_net_input_dim = opt.hidden_size * 2
            elif opt.variance_memory_type in [1, 3, 5]:
                recog_net_input_dim = opt.hidden_size

            if opt.variance_memory_type in [0, 1, 4, 5]:
                variance_infer_out_dim = opt.latent_size * 2
            elif opt.variance_memory_type in [2, 3]:
                variance_infer_out_dim = opt.latent_size

            self.recog_net = nn.Sequential(nn.Linear(recog_net_input_dim, opt.latent_size),
                                           nn.Linear(opt.latent_size, variance_infer_out_dim))

            # # encode
            # self.fc1  = nn.Linear(x_feat_size + c_feat_size, latent_size)
            # self.fc21 = nn.Linear(latent_size, latent_size)

            # Prior Network
            self.prior_net = nn.Sequential(nn.Linear(opt.hidden_size, opt.hidden_size),
                                           nn.Linear(opt.latent_size, variance_infer_out_dim))

            # decode
            self.fc31 = nn.Linear(opt.latent_size + opt.hidden_size, opt.latent_size)
            self.fc41 = nn.Linear(opt.latent_size, opt.hidden_size)
            self.fc32 = nn.Linear(opt.latent_size + opt.hidden_size, opt.latent_size)
            self.fc42 = nn.Linear(opt.latent_size, opt.hidden_size)

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        if self.opt.src_tgt_latent_merge_type in [2] and self.opt.vae_type in [6, 7]:
            self.fc1 = nn.Linear(opt.latent_size * 2, opt.latent_size)
            if self.opt.vae_type in [7]:
                self.fc2 = nn.Linear(opt.latent_size * 2, opt.latent_size)
            self.relu = nn.ReLU()

    def merge_src_tgt_forword(self, src, tgt):
        if not (self.opt.src_tgt_latent_merge_type in [2] and self.opt.vae_type in [6, 7]):
            return src # or return None

        #print("src shape:", src.shape)
        #print("tgt shape:", tgt.shape)
        src_tgt = torch.cat((src, tgt), 2)
        #print("src_tgt shape:", src_tgt.shape)
        output = self.fc1(src_tgt)
        output_nonlinear = self.relu(output)
        return output_nonlinear

    def merge_src_tgt_forword_memory(self, src, tgt):
        if not (self.opt.src_tgt_latent_merge_type in [2] and self.opt.vae_type in [7]):
            return src # or return None

        src_tgt = torch.cat((src, tgt), 2)
        output = self.fc2(src_tgt)
        output_nonlinear = self.relu(output)
        return output_nonlinear

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def concate_and_decode(self, c, z): # P(x|z, c)
        '''
        z: (bs, latent_size) latent z of autoencoder, target(recog or prior)
        c: (bs, hidden_size) condition, source
        '''
        if 1 == self.opt.source_dropout_type and 0 != self.opt.source_dropout_rate \
            and random.random() < self.opt.source_dropout_rate and self.training:
            output = self.sigmoid(self.fc41(z))
        else:
            inputs = torch.cat([c, z], -1) # (bs, latent_size+class_size)
            h3 = self.relu(self.fc31(inputs))
            output = self.sigmoid(self.fc41(h3))
        return output

    def concate_and_decode_prior(self, c, z): # P(x|z, c)
        '''
        z: (bs, latent_size) latent z of autoencoder, target(recog or prior)
        c: (bs, hidden_size) condition, source
        '''
        if 1 == self.opt.source_dropout_type and 0 != self.opt.source_dropout_rate \
            and random.random() < self.opt.source_dropout_rate and self.training:
            output = self.sigmoid(self.fc42(z))
        else:
            inputs = torch.cat([c, z], -1) # (bs, latent_size+class_size)
            h3 = self.relu(self.fc32(inputs))
            output = self.sigmoid(self.fc42(h3))
        return output

    def source_dropout_to_zero(self, src):
        if 0 == self.opt.source_dropout_rate:
            return src

        batch_size = src.shape[1]
        dim = src.shape[2]

        droped = []
        dropout_filter = Variable(torch.ones(src.shape), requires_grad=False)
        for i in range(batch_size):
            if random.random() < self.opt.source_dropout_rate:
                dropout_filter[0][i] = Variable(torch.zeros(dim))
                droped.append(i)
        dropout_filter = dropout_filter.type('torch.FloatTensor')
        if self.opt.use_cuda:
            dropout_filter = dropout_filter.cuda()
        #print("dropout_filter type:", dropout_filter.type())
        #print("src type:", src.type())
        new_src = src * dropout_filter
        #print("droped:", " ".join(list(map(str, droped))))
        return new_src

    def inference(self, src, mem_tgt):
        if not self.opt.vae_type in [8]:
            return

        prior_tgt = self.prior_net(mem_tgt)

        if self.opt.variance_memory_type in [0, 1, 4, 5]:
            prior_mu, prior_logvar = torch.split(prior_tgt, self.opt.latent_size, dim=-1)
            prior_z = self.reparametrize(prior_mu, prior_logvar)
        elif self.opt.variance_memory_type in [2, 3]:
            prior_z = prior_tgt

        prior_z_with_src = self.concate_and_decode_prior(src, prior_z)
        return prior_z_with_src

    def inference_by_posterior(self, src, tgt, memory_tgt):
        self.training = False
        recog_z_with_src, prior_z_with_src, recog_mu, recog_logvar, prior_mu, prior_logvar \
            = self.forward(src, tgt, memory_tgt)
        return recog_z_with_src

    def forward(self, src, tgt, memory_tgt):
        if not self.opt.vae_type in [8]:
            return

        recog_z_with_src, prior_z_with_src, recog_mu, recog_logvar, prior_mu, prior_logvar = [None] * 6
        if not memory_tgt is None:
            prior_tgt = self.prior_net(memory_tgt)

        if self.opt.variance_memory_type in [0, 2, 4]:
            if memory_tgt is None:
                print("Error. memory_tgt is None cannot support opt.variance_memory_type in [0, 2, 4]. exit")
                exit(-1)
            tgt_mem_tgt = torch.cat((tgt, memory_tgt), 2)
            recog_tgt = self.recog_net(tgt_mem_tgt)
        elif self.opt.variance_memory_type in [1, 3, 5]:
            recog_tgt = self.recog_net(tgt)

        if self.opt.variance_memory_type in [0, 1, 4, 5]:
            recog_mu, recog_logvar = torch.split(recog_tgt, self.opt.latent_size, dim=-1)
            if self.opt.only_rm_variantional_path == 0:
                recog_z = self.reparametrize(recog_mu, recog_logvar)
            elif self.opt.only_rm_variantional_path == 1:
                recog_z = recog_mu
            if not memory_tgt is None:
                prior_mu, prior_logvar = torch.split(prior_tgt, self.opt.latent_size, dim=-1)
                if self.opt.only_rm_variantional_path == 0:
                    prior_z = self.reparametrize(prior_mu, prior_logvar)
                elif self.opt.only_rm_variantional_path == 1:
                    prior_z = prior_mu
        elif self.opt.variance_memory_type in [2, 3]:
            recog_z = recog_tgt
            if not memory_tgt is None:
                prior_z = prior_tgt

        if 0 == self.opt.source_dropout_type and 0 != self.opt.source_dropout_rate:
            src = self.source_dropout_to_zero(src)

        recog_z_with_src = self.concate_and_decode(src, recog_z) # param can be shared (concate_and_decode and concate_and_decode_prior)
        if self.opt.variance_memory_type in [2, 3, 4, 5] and (not memory_tgt is None):
            prior_z_with_src = self.concate_and_decode_prior(src, prior_z)
        else:
            prior_z_with_src = None

        return recog_z_with_src, prior_z_with_src, recog_mu, recog_logvar, prior_mu, prior_logvar

class CVAE(nn.Module):
    def __init__(self, x_feat_size, c_feat_size, latent_size):
        super(CVAE, self).__init__()
        self.x_feat_size = x_feat_size
        self.c_feat_size = c_feat_size
        self.latent_size = latent_size
        # Recognition Network
        self.recog_net = nn.Sequential(nn.Linear(x_feat_size + c_feat_size, latent_size),
                                       nn.Linear(latent_size, latent_size*2))

        # # encode
        # self.fc1  = nn.Linear(x_feat_size + c_feat_size, latent_size)
        # self.fc21 = nn.Linear(latent_size, latent_size)
        # self.fc22 = nn.Linear(latent_size, latent_size)

        # Prior Network
        self.prior_net = nn.Sequential(nn.Linear(c_feat_size, latent_size),
                                       nn.Linear(latent_size, latent_size*2))

        # decode
        self.fc3 = nn.Linear(latent_size + c_feat_size, latent_size)
        self.fc4 = nn.Linear(latent_size, x_feat_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def recog_encode(self, x, c): # Q(z|x, c)
        inputs = torch.cat([x, c], -1)

        recog_mulogvar = self.recog_net(inputs)
        recog_mu, recog_logvar = torch.split(recog_mulogvar,self.latent_size, dim=-1)
        return recog_mu, recog_logvar

    def prior_encode(self, c):

        prior_mulogvar = self.prior_net(c)
        prior_mu, prior_logvar = torch.split(prior_mulogvar,self.latent_size, dim=-1)

        return prior_mu, prior_logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], -1) # (bs, latent_size+class_size)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def inference(self, c):
        prior_mu, prior_logvar = self.prior_encode(c)
        prior_z = self.reparametrize(prior_mu, prior_logvar)
        return self.decode(prior_z, c)

    def forward(self, x, c):
        recog_mu, recog_logvar = self.recog_encode(x, c)
        prior_mu, prior_logvar = self.prior_encode(c)
        recog_z = self.reparametrize(recog_mu, recog_logvar)
        # prior_z = self.reparametrize(prior_mu, prior_logvar)
        return self.decode(recog_z, c), recog_mu, recog_logvar, prior_mu, prior_logvar

    def forward_with_prior(self, x, c):
        recog_mu, recog_logvar = self.recog_encode(x, c)
        prior_mu, prior_logvar = self.prior_encode(c)
        recog_z = self.reparametrize(recog_mu, recog_logvar)
        prior_z = self.reparametrize(prior_mu, prior_logvar)
        return self.decode(recog_z, c), self.decode(prior_z, c), recog_mu, recog_logvar, prior_mu, prior_logvar

#class DataCluster(nn.Module):
class DataCluster():
    def __init__(self, opt):
        #super(DataCluster, self).__init__()
        # current data for next clustering
        self.src_samples = []
        self.tgt_samples = []
        self.opt = opt

        # model data for NN training
        self.cluster_centers_key = None # cpu data, detach data(no auto grad)
        self.cluster_centers_value = None # cpu data, detach data(no auto grad)

        # sample level model data for NN training
        self.last_src_samples = []
        self.last_tgt_samples = []
        self.cluster2samples = {}
        self.sample2cluster = []
        self.cluster_list = []

        # options
        self.sample_rate = opt.training_data_sample_rate_for_cluster 

        # handles
        self.time = time_utils.StatRunTime()
        self.mean_square_loss = torch.nn.MSELoss()

    def append_one_pair(self, src_vector, tgt_vector):
        self.src_samples.append(src_vector.detach().cpu().numpy())
        self.tgt_samples.append(tgt_vector.detach().cpu().numpy())
        if self.opt.debug_mode >= 6:
            print("len(src_samples):", len(self.src_samples))

    def append_by_prob(self, batch_src_vectors, batch_tgt_vectors): 
        # batch_src_vectors and batch_tgt_vectors is torch.Size([num_of_layer, batch_size, latent_dim])
        batch_src_vectors = batch_src_vectors[0]
        batch_tgt_vectors = batch_tgt_vectors[0]
        for src_vector, tgt_vector in zip(batch_src_vectors, batch_tgt_vectors):
            if random.random() < self.sample_rate:
                self.append_one_pair(src_vector, tgt_vector)

    def update_cluster_labels(self, kmeans_model):
        self.cluster2samples = {}
        self.sample2cluster = kmeans_model.labels_
        if self.opt.debug_mode >= 5:
            print("len(kmeans_model.labels_):", len(kmeans_model.labels_))
            print("len(kmeans_model.cluster_centers_):", len(kmeans_model.cluster_centers_))
        self.cluster_list = np.unique(kmeans_model.labels_)
        for sample, cluster in enumerate(self.sample2cluster):
            if not cluster in self.cluster2samples:
                self.cluster2samples[cluster] = []
            self.cluster2samples[cluster].append(sample)

    def aggregate_cluster_centers(self, kmeans_model):
        cluster_centers_key_np = kmeans_model.cluster_centers_
        self.update_cluster_labels(kmeans_model)
        cluster_size = len(cluster_centers_key_np)
        vector_dim = len(cluster_centers_key_np[0])
        cluster_centers_value_np = np.zeros([cluster_size, vector_dim])

        for cluster in self.cluster_list:
            if not cluster in self.cluster2samples:
                continue
            sample_num = len(self.cluster2samples[cluster])
            for sample in self.cluster2samples[cluster]:
                if self.opt.debug_mode >= 7:
                    print("cluster:", cluster, "sample:", sample, "len(cluster_centers_value_np):", len(cluster_centers_value_np), "len(tgt_samples):", len(self.tgt_samples))
                cluster_centers_value_np[cluster] = \
                    np.add(cluster_centers_value_np[cluster], self.tgt_samples[sample])
            cluster_centers_value_np[cluster] = cluster_centers_value_np[cluster] / sample_num

        self.cluster_centers_key = Variable(torch.tensor(cluster_centers_key_np).float(), requires_grad = False)
        self.cluster_centers_value = Variable(torch.tensor(cluster_centers_value_np).float(), requires_grad = False)
        #self.cluster_centers_key = torch.tensor(cluster_centers_key_np).float() # Not sure whether it has gradient? may not. Notice!!!
        #self.cluster_centers_value = torch.tensor(cluster_centers_value_np).float() # Not sure whether it has gradient? may not. Notice!!!
        if 1 == self.opt.cluster_param_in_cuda:
            self.cluster_centers_key = self.cluster_centers_key.cuda()
            self.cluster_centers_value = self.cluster_centers_value.cuda()

    def clear_samples(self):
        self.last_src_samples = self.src_samples
        self.last_tgt_samples = self.tgt_samples
        self.src_samples = []
        self.tgt_samples = []


    def save_cluster_txt(self, epoch, with_training_sample = False, mark_on_name = ""):
        if self.cluster_centers_key is None or self.cluster_centers_value is None:
            print("attempt to save cluster center but it is None. quit save_cluster_txt and continue")
            return

        if "" == mark_on_name:
            filename = self.opt.out_dir + "/data_cluster" + str(epoch) + ".txt"
        else:
            filename = self.opt.out_dir + "/data_cluster" + str(epoch) + "_" + mark_on_name + ".txt"
        if self.opt.debug_mode >= 2:
            print('save txt model from %s...'%(filename))
        f = open(filename, "w")

        out_in_dim_list = []
        if 1 == self.opt.cluster_param_in_cuda:
            self.cluster_centers_key = self.cluster_centers_key.cpu()
            self.cluster_centers_value = self.cluster_centers_value.cpu()
        out_in_dim = data_utils.save_2Dnumpy_as_list_to_txt(f, self.cluster_centers_key.numpy())
        out_in_dim_list.append(out_in_dim)
        out_in_dim = data_utils.save_2Dnumpy_as_list_to_txt(f, self.cluster_centers_value.numpy())
        out_in_dim_list.append(out_in_dim)
        if 1 == self.opt.cluster_param_in_cuda:
            self.cluster_centers_key = self.cluster_centers_key.cuda()
            self.cluster_centers_value = self.cluster_centers_value.cuda()
    
        if with_training_sample:
            if 0 != len(self.src_samples) \
                    and 0 != len(self.tgt_samples):
                src_samples = self.src_samples
                tgt_samples = self.tgt_samples
            elif 0 != len(self.last_src_samples) \
                    and 0 != len(self.last_tgt_samples):
                src_samples = self.last_src_samples
                tgt_samples = self.last_tgt_samples
            else: # means first epoch and did not do clear_samples
                pass
            
            out_in_dim = data_utils.save_2Dnumpy_as_list_to_txt(f, src_samples)
            out_in_dim_list.append(out_in_dim)
            out_in_dim = data_utils.save_2Dnumpy_as_list_to_txt(f, tgt_samples)
            out_in_dim_list.append(out_in_dim)

        print("save_cluster_txt done. outdim and indim list:", out_in_dim_list)

        f.close()

    def load_cluster_txt(self, epoch, with_training_sample = False):
        filename = self.opt.out_dir + "/data_cluster" + str(epoch) + ".txt"
        if self.opt.debug_mode >= 2:
            print('load txt model from %s...'%(filename))
        try:
            f = open(filename, "r")
        except:
            print("Error Or Warning! open file in load_cluster_txt failed. filename:", \
                filename, "Ignore this message if do not use inital cluster.")
            return

        data_list = []
        data = []
        outdim, indim = [-1, -1]
        out_in_dim_list = []
        for line in f:
            slots = line.strip("\n").split("\t")
            if 2 == len(slots): # new data comes
                if 0 != len(out_in_dim_list): # not first data part
                    if len(data) != out_in_dim_list[-1][0]:
                        print("error.len(data)!=outdim. len(data):",\
                            len(data), "outdim:", out_in_dim_list[-1][0])
                        return
                        if 0 != len(data):
                            if len(data[0]) != out_in_dim_list[-1][1]:
                                print("error.len(data[0])!=indim. len(data[0])",\
                                    len(data), "indim:", out_in_dim_list[-1][1]) 
                                return

                    data_list.append(data)
                    data = []
                out_in_dim_list.append(list(map(int, slots)))
            if 1 == len(slots): # new line for old data
                vec = list(map(float, slots[0].split()))
                data.append(vec)
        f.close()

        if 2 > len(data_list):
            print("error. 2 > len(data_list). len(data_list):", len(data_list))

        self.cluster_centers_key = \
            Variable(torch.tensor(np.array(data_list[0])).float(), \
                    requires_grad = False)
        self.cluster_centers_value = \
            Variable(torch.tensor(np.array(data_list[1])).float(), \
                    requires_grad = False)
        if 1 == self.opt.cluster_param_in_cuda:
            self.cluster_centers_key = self.cluster_centers_key.cuda()
            self.cluster_centers_value = self.cluster_centers_value.cuda()

        if with_training_sample:
            if 4 > len(data_list):
                print("warning. 2 > len(data_list). len(data_list):", len(data_list))
            else:
                self.src_samples = np.array(data_list[2])
                self.tgt_samples = np.array(data_list[3])

        print("load_cluster_txt done. outdim and indim list:", out_in_dim_list)
        return

    def print_running_time(self):
        #if random.random() > 0.01:
        #    return

        self.time.print_time("before_gpu_to_cpu", "gpu_to_cpu_done")
        self.time.print_time("gpu_to_cpu_done", "matmul_done")
        self.time.print_time("matmul_done", "softmax_done")
        self.time.print_time("softmax_done", "weight_sum_done")
        self.time.print_time("weight_sum_done", "cpu_to_gpu_done")
        self.time.print_time("begin_forward", "end_forward")
        self.time.print_time("before_cluster_loss", "cluster_loss_done")

        self.time.clear_all_bufs()

    def forward(self, src_latent_z):
        ''' src_latent_z must be (batch_size * latent_dim) '''
        self.time.get_time("begin_forward")
        if self.cluster_centers_key is None and self.cluster_centers_value is None:
            #print("pass data_cluster forward due to no cluster")
            return None

        # Alpha_src = ClusterKeyMatrix * Z_src (dot product between Z_src and each row of ClusterKeyMatrix)
        # cluster_centers_key(cluster_size*latent_dim) src_latent_z(batch_size*latent_dim) alpha_src(batch_size*cluster_size)
        # key_mul_z = self.cluster_centers_key * src_latent_z # elelment-wise product 
        self.time.get_time("before_gpu_to_cpu")
        if 0 == self.opt.cluster_param_in_cuda and src_latent_z.is_cuda:
            src_latent_z = src_latent_z.cpu()
            if self.opt.debug_mode >= 7:
                    print("src_latent_z gpu -> cpu")
        self.time.get_time("gpu_to_cpu_done")

        if 0 == self.opt.get_memory_target_type:
            alpha_src = src_latent_z.matmul(torch.t(self.cluster_centers_key))
            self.time.get_time("matmul_done")
            alpha_src_norm = torch.nn.functional.softmax(alpha_src, dim=1)
            self.time.get_time("softmax_done")
            alpha_tgt_norm = alpha_src_norm
            memory_tgt = operation_utils.matrix_weight_sum_by_batch_vectors(self.cluster_centers_value, alpha_tgt_norm)
            self.time.get_time("weight_sum_done")
        elif 1 == self.opt.get_memory_target_type:
            max_sim_col, max_sim = \
                operation_utils.get_vectorwise_nearest_neighbor_of_two_matrix(\
                src_latent_z, self.cluster_centers_key, \
                self.opt.sim_type_for_memory_search, filter_totally_matched_sample=True, \
                neighbourhood_of_filter=self.opt.similarity_threshold_for_itself)
            self.time.get_time("search most similar candidate done")
            #print("cluster_centers_value shape:", self.cluster_centers_value.shape)
            memory_tgt = \
                operation_utils.fetch_vec_from_matrix_by_index(\
                    self.cluster_centers_value, max_sim_col)
            #print("memory_tgt:", memory_tgt)
            #print("memory_tgt.shape:", memory_tgt.shape)
            self.time.get_time("fetch most similar candidate done")
            #print("max_sim_col:\t", print_utils.numpy_array_to_str(max_sim_col))
            #print("max_sim:\t", print_utils.numpy_array_to_str(max_sim.detach().cpu().numpy()))
            #print("max_sim_col2:\t", print_utils.numpy_array_to_str_with_index(max_sim_col))
            #print("max_sim2:\t", print_utils.numpy_array_to_str_with_index(max_sim.detach().cpu().numpy()))

            
        if not memory_tgt.is_cuda:
            memory_tgt = memory_tgt.cuda()
            if self.opt.debug_mode >= 7:
                print("memory_tgt cpu -> gpu")
        self.time.get_time("cpu_to_gpu_done")
        self.time.get_time("end_forward")
        return memory_tgt

    def get_loss(self, target_latent_z, memory_target_z):
        ''' see https://pytorch.org/docs/master/nn.html#torch.nn.MSELoss '''
        self.time.get_time("before_cluster_loss")
        msq_loss_output = self.mean_square_loss(target_latent_z, memory_target_z)
        #cos = F.cosine_similarity(target_latent_z, memory_target_z, dim=-1)
        #print("cos of target and target^:", cos)
        #print("cos of target and target^ 2:\t", print_utils.numpy_array_to_str(cos.detach().cpu().numpy()))
        #print("cos of target and target^ 3:\t", print_utils.numpy_array_to_str_with_index(cos.detach().cpu().numpy()))
        self.time.get_time("cluster_loss_done")
        return msq_loss_output

class CvaeDialog(nn.Module):
    def __init__(self, x_encoder, c_encoder, decoder, cvae, gmm_net, latent_net, generator, data_cluster, opt):
        super(CvaeDialog, self).__init__()
        self.x_encoder = x_encoder
        self.c_encoder = c_encoder
        self.cvae = cvae
        self.gmm_net = gmm_net
        self.decoder = decoder
        self.generator = generator
        self.latent_net = latent_net
        self.data_cluster = data_cluster
        self.opt = opt
        self.time = time_utils.StatRunTime()
        self.layers_map_for_model = {0:["c_encoder_dict","decoder_dict","generator_dict"], \
            1:["x_encoder_dict","c_encoder_dict","cvae_dict","decoder_dict","generator_dict"], \
            2:["x_encoder_dict","c_encoder_dict","cvae_dict","decoder_dict","generator_dict"], \
            3:["x_encoder_dict","c_encoder_dict","gmm_dict","decoder_dict","generator_dict"], \
            4:["x_encoder_dict","c_encoder_dict","gmm_dict","decoder_dict","generator_dict"], \
            5:["c_encoder_dict","decoder_dict","generator_dict"], \
            6:["x_encoder_dict","c_encoder_dict","latent_dict","decoder_dict","generator_dict"], \
            7:["x_encoder_dict","c_encoder_dict","latent_dict","decoder_dict","generator_dict"], \
            8:["x_encoder_dict","c_encoder_dict","latent_dict","decoder_dict","generator_dict"]}

    def set_paramater_to_cuda(self):
        self.x_encoder = self.x_encoder.cuda()
        self.c_encoder = self.c_encoder.cuda()
        self.cvae = self.cvae.cuda()
        self.gmm_net = self.gmm_net.cuda()
        self.decoder = self.decoder.cuda()
        self.generator = self.generator.cuda()
        self.latent_net = self.latent_net.cuda()

    def print_latent_representation(self, x, z, type):
        if 0 == self.opt.save_z_and_sample:
            return 
        if input is None:
            return
        if self.opt.debug_mode >= 7:
            print("x size:", x.size())
            print("x:", x)
            print("z size:", z.size())
            print("z:", z)
            print("z[0] size:", z[0].size())
            print("z[0]:", z[0])
        x_t = torch.transpose(x, 0, 1)
        if self.opt.debug_mode >= 7:
            print("x_t size:", x_t.size())
            print("x_t:", x_t)
        print_utils.print_matrix_with_text(z[0], x_t, self.opt.variable_src_dict.itos, 6, sys.stderr, type)

    def freeze_modules(self, modules_to_freeze):
        """
            modules_to_freeze is a list to describe which module to be freezed
                e.g. "x_encoder,decoder"
        """

        if self.opt.debug_mode >= 3:
            print("Before freeze_modules")
            print_utils.print_nn_module_model(self)

        modules_to_freeze = modules_to_freeze.split(",")
        print("self._modules:", self._modules)
        print("modules_to_freeze:", modules_to_freeze)
        for fm in modules_to_freeze:
            if not fm in self._modules:
                print("error. not fm in self._modules. fm:", fm)
                continue
            for p in self._modules[fm].parameters():
                p.requires_grad = False

        if self.opt.debug_mode >= 3:
            print("After freeze_modules")
            print_utils.print_nn_module_model(self, 2)
            """
            for i, m in enumerate(self._modules.keys()):
                print("i:", i, "module name:", m)
                for j, l in enumerate(self._modules[m]._modules.keys()):
                    print("\tj:", j, "layer name:", l, "layer:", self._modules[m]._modules[l])
                    for k, p in enumerate(self._modules[m]._modules[l].parameters()):
                        print("\t\tk:", k, "parameter shape:", p.shape, "parameter requires_grad:", p.requires_grad)
            """

    def order_sequence(self, inputs, lengths):
        idx_len_pair = []
        for i,l in enumerate(lengths):
            idx_len_pair.append((i,l))
        sorted_idx_len_pair=sorted(idx_len_pair,key=lambda x:x[1],reverse=True)
        sorted_idx = []
        sorted_length = []
        for x in sorted_idx_len_pair:
            sorted_idx.append(x[0])
            sorted_length.append(x[1])

        order_inputs = inputs[:,sorted_idx]
        return order_inputs, sorted_length, sorted_idx

    def reorder_enc_seqence(self, inputs, sorted_idx):
        raw_sorted_pair = []
        for raw_idx,sorted_idx in enumerate(sorted_idx):
            raw_sorted_pair.append((raw_idx,sorted_idx))
        sorted_pair=sorted(raw_sorted_pair,key=lambda x:x[1],reverse=False)
        raw_indices = [x[0] for x in sorted_pair]
        #print("in error. inputs:", inputs)
        #print("in error. raw_indices:", raw_indices)
        #reorder_inputs = np.array(inputs)[:,raw_indices,:]
        #reorder_inputs = reorder_inputs.tolist()
        reorder_inputs = inputs[:,raw_indices,:]
        #print("in error. reorder_inputs:", reorder_inputs)
        #print("in error. inputs.size():", inputs.size())
        return reorder_inputs

    def x_encode(self, input, lengths=None, hidden=None): #Feed input to normal encoder and get output
        #print("input:", input)
        order_inputs, sorted_length, sorted_idx =self.order_sequence(input,lengths)
        enc_outputs, enc_hidden = self.x_encoder(order_inputs, sorted_length, None)
        #print("enc_hidden:",enc_hidden)
        #print("enc_hidden[0].size:",enc_hidden[0].size())
        #print("enc_hidden[1].size:",enc_hidden[1].size())
        #print("sorted_idx:",sorted_idx)
        enc_hidden = self.reorder_enc_seqence(enc_hidden,sorted_idx)
        enc_outputs = self.reorder_enc_seqence(enc_outputs,sorted_idx)
        self.print_latent_representation(input, enc_hidden, "input")
        #if self.opt.save_z_and_sample:
        #    if not input is None:
        #        print("input size:", input.size())
        #        print("input:", input)
        #        print("enc_hidden size:", enc_hidden.size())
        #        print("enc_hidden:", enc_hidden)
        #        print("enc_hidden[0] size:", enc_hidden[0].size())
        #        print("enc_hidden[0]:", enc_hidden[0])
        #        input_t = torch.transpose(input, 0, 1)
        #        print("input_t size:", input_t.size())
        #        print("input_t:", input_t)
        #        print_utils.print_matrix_with_text(enc_hidden[0], input_t, self.opt.variable_src_dict.itos, 4, sys.stderr, "input")
        return enc_outputs, enc_hidden

    def c_encode(self, input, lengths=None, hidden=None): #Feed input to normal encoder and get output
        order_inputs, sorted_length, sorted_idx =self.order_sequence(input,lengths)
        #print("order_inputs in c_encode:", order_inputs)
        #print("order_inputs[0][0.type] in c_encode:", order_inputs[0][0].type())
        #print("sorted_length in c_encode:", sorted_length)
        enc_outputs, enc_hidden = self.c_encoder(order_inputs, sorted_length, None)
        enc_hidden = self.reorder_enc_seqence(enc_hidden,sorted_idx)
        enc_outputs = self.reorder_enc_seqence(enc_outputs,sorted_idx)
        self.print_latent_representation(input, enc_hidden, "condition")
        return enc_outputs, enc_hidden

    def decode(self, input, context, state):
        dec_outputs , dec_hiddens, attn = self.decoder(
                input, context, state
            )     

        return dec_outputs, dec_hiddens, attn

    def forward(self, src_inputs, src_lengths, tgt_inputs, tgt_lengths, only_get_samples_for_cluster=False): # Main Step: connect all components together
        dec_outputs, attn, recog_mu, recog_logvar, prior_mu, prior_logvar, P_c_given_x, gmm_loss, loss_for_clustering, dec_outputs_memory, attn_memory = [None] * 11

        # Encoder
        self.time.get_time("encoder_start")
        if self.opt.vae_type in [1, 2, 4, 5, 6, 7, 8]:
            x_enc_outputs, x_enc_hidden = self.x_encode(tgt_inputs, tgt_lengths) # Normal encoder
        if self.opt.vae_type in [0, 1, 2, 3, 4, 6, 7, 8]:
            c_enc_outputs, c_enc_hidden = self.c_encode(src_inputs, src_lengths) # Normal encoder
        self.time.get_time("encoder_done")

        if self.opt.vae_type in [1, 2, 4, 6, 7, 8]:
            print_utils.save_latent_Z_with_text(src_inputs, \
                    c_enc_hidden[0], tgt_inputs, x_enc_hidden[0], \
                    self.opt, 6, sys.stderr, "src_tgt")
        elif self.opt.vae_type in [0]:
            print_utils.save_latent_Z_with_text(src_inputs, \
                    c_enc_hidden[0], tgt_inputs, None, \
                    self.opt, 6, sys.stderr, "src_tgt")
        elif self.opt.vae_type in [5]:
            print_utils.save_latent_Z_with_text(src_inputs, \
                    None, tgt_inputs, x_enc_hidden[0], \
                    self.opt, 6, sys.stderr, "src_tgt")

        # Cluster and save for clustering
        if self.opt.vae_type in [6, 7, 8]:
            if 0 != self.opt.training_data_sample_rate_for_cluster and \
                    self.opt.vae_type in [1, 2, 4, 6, 7, 8]:
                if 0 == self.opt.train_collect_clusterdata_sync:
                    if only_get_samples_for_cluster:
                        self.data_cluster.sample_rate = 1
                        self.data_cluster.append_by_prob(c_enc_hidden, x_enc_hidden)
                else:
                    self.data_cluster.append_by_prob(c_enc_hidden, x_enc_hidden)
                    
                self.time.get_time("cluster_append_data_done")
                if only_get_samples_for_cluster:
                    return None

            if self.opt.vae_type in [7, 8] and 0 >= self.opt.cluster_num:
                print("Error. self.opt.vae_type in [7,8] and 0 >= self.opt.cluster_num is not allowed!")
                exit(-1)
            x_enc_hidden_memory = None
            if 0 < self.opt.cluster_num: # means use clustering 
                memory_target_z = self.data_cluster.forward(c_enc_hidden[0])
                if not memory_target_z is None:
                    loss_for_clustering = self.data_cluster.get_loss(x_enc_hidden[0], memory_target_z)
                    x_enc_hidden_memory = x_enc_hidden.clone().detach()
                    memory_target_z = memory_target_z.detach()
                    x_enc_hidden_memory[0] = memory_target_z # !!! important
                    #x_enc_hidden = memory_target_z # !!! important
                else:
                    if self.opt.vae_type in [6]:
                        x_enc_hidden_memory = x_enc_hidden
                    if self.opt.vae_type in [7, 8] and \
                        1 == self.opt.s2saememory_mode_before_first_cluster:
                        x_enc_hidden_memory = x_enc_hidden

        if self.opt.debug_mode >= 7:
            if c_enc_hidden.is_cuda:
                print("c_enc_hidden is still in GPU after data_cluster.forward")
            else:
                print("c_enc_hidden is not in GPU after data_cluster.forward")
        self.time.get_time("cluster_done")

        # Latent
        if self.opt.vae_type in [1, 2]:
            if 1 == self.opt.cvae_print_reconstruct_loss and self.opt.vae_type in [1]:
                latent_vector_z, latent_vector_z_prior, recog_mu, recog_logvar, prior_mu, prior_logvar \
                    = self.cvae.forward_with_prior(x_enc_hidden, c_enc_hidden) # latent representation part. NN on encoder output and get z
            else:
                latent_vector_z, recog_mu, recog_logvar, prior_mu, prior_logvar \
                    = self.cvae(x_enc_hidden, c_enc_hidden) # latent representation part. NN on encoder output and get z
        if self.opt.vae_type in [3, 4]:
            latent_vector_z, P_c_given_x, gmm_loss = self.gmm_net(c_enc_hidden, src_inputs, tgt_inputs)
        if self.opt.vae_type in [6]:
            if 0 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = torch.cat((c_enc_hidden, x_enc_hidden_memory), 2)
                assert(self.opt.use_src_or_tgt_attention in [3])
            elif 1 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = c_enc_hidden + x_enc_hidden_memory
            elif 2 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = self.latent_net.merge_src_tgt_forword(c_enc_hidden, x_enc_hidden_memory)
        if self.opt.vae_type in [7]:
            if 0 == self.opt.src_tgt_latent_merge_type:
                #print("c_enc_hidden shape:", c_enc_hidden.shape)
                #print("x_enc_hidden shape:", x_enc_hidden.shape)
                latent_vector_z = torch.cat((c_enc_hidden, x_enc_hidden), 2)
                if not x_enc_hidden_memory is None:
                    latent_vector_z_memory = torch.cat((c_enc_hidden, x_enc_hidden_memory), 2)
                #print("latent_vector_z shape:", latent_vector_z.shape)
                assert(self.opt.use_src_or_tgt_attention in [3])
            elif 1 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = c_enc_hidden + x_enc_hidden
                if not x_enc_hidden_memory is None:
                    latent_vector_z_memory = c_enc_hidden + x_enc_hidden_memory
            elif 2 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = self.latent_net.merge_src_tgt_forword(c_enc_hidden, x_enc_hidden)
                if not x_enc_hidden_memory is None:
                    latent_vector_z_memory = self.latent_net.merge_src_tgt_forword_memory(c_enc_hidden, x_enc_hidden_memory)
        if self.opt.vae_type in [8]:
            latent_vector_z, latent_vector_z_memory, \
                recog_mu, recog_logvar, \
                prior_mu, prior_logvar \
                = self.latent_net.forward(\
                    c_enc_hidden, x_enc_hidden, x_enc_hidden_memory)
            #if self.opt.variance_memory_type in [2, 3, 4, 5] and (latent_vector_z_memory is None):
            #    print("Warning. self.opt.variance_memory_type in [2, 3, 4, 5] and (latent_vector_z_memory is None")
            #if self.opt.variance_memory_type in [0,1,4,5] and (recog_mu is None or recog_logvar is None or prior_mu is None or prior_logvar is None):
            #    print("Warning. self.opt.variance_memory_type in [0,1,4,5] and (mu / var is None")
        self.time.get_time("latent_done")
 
 
        # Decoder
        if self.opt.vae_type in [1, 2, 3, 4]: # c_enc_hidden is z, c_enc_outputs is all hidden states over whole sequence
            dec_outputs, dec_hiddens, attn = self.decode(tgt_inputs,c_enc_outputs,latent_vector_z) # Normal decoder
            if self.opt.vae_type in [1] and 1 == self.opt.cvae_print_reconstruct_loss:
                dec_outputs_memory, dec_hiddens_memory, attn_memory = self.decode(tgt_inputs, c_enc_outputs, latent_vector_z_prior)
        if 0 == self.opt.vae_type:
            #self.print_latent_representation(src_inputs, c_enc_hidden, "src")
            dec_outputs, dec_hiddens, attn = self.decode(tgt_inputs,c_enc_outputs,c_enc_hidden)
        if 5 == self.opt.vae_type:
            #self.print_latent_representation(tgt_inputs, x_enc_hidden, "tgt")
            dec_outputs, dec_hiddens, attn = self.decode(tgt_inputs,x_enc_outputs,x_enc_hidden)
        if self.opt.vae_type in [6, 7, 8]:
            if 0 == self.opt.use_src_or_tgt_attention:
                states_as_attention = c_enc_outputs
            elif 1 == self.opt.use_src_or_tgt_attention:
                states_as_attention = x_enc_outputs
            elif 2 == self.opt.use_src_or_tgt_attention: 
                states_as_attention = c_enc_outputs + x_enc_outputs
            elif 3 == self.opt.use_src_or_tgt_attention:
                #print("c_enc_outputs shape:", c_enc_outputs.shape)
                #print("x_enc_outputs shape:", x_enc_outputs.shape)
                states_as_attention = torch.cat((c_enc_outputs, x_enc_outputs), 2)
                #print("states_as_attention shape:", states_as_attention.shape)
            dec_outputs, dec_hiddens, attn = self.decode(tgt_inputs, states_as_attention, latent_vector_z)
            if self.opt.vae_type in [7, 8] and (not x_enc_hidden_memory is None) \
                and (not latent_vector_z_memory is None):
                dec_outputs_memory, dec_hiddens_memory, attn_memory = self.decode(tgt_inputs, states_as_attention, latent_vector_z_memory)
        self.time.get_time("decoder_done")

        if random.random() < 0.01 and self.opt.debug_mode >= 4:
            self.data_cluster.print_running_time()
            self.print_running_time()

        return dec_outputs, attn, recog_mu, recog_logvar, prior_mu, prior_logvar, P_c_given_x, gmm_loss, loss_for_clustering, dec_outputs_memory, attn_memory

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'model_dict': self.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):   
        print('Loding model from %s...'%(filename))
        #print("opt.use_cuda:", opt.use_cuda)
        if self.opt.use_cuda:
            ckpt = torch.load(filename, map_location=torch.device(self.opt.gpuid))
        else:
            ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
            #device = torch.device('cpu')
            #ckpt = torch.load(filename, map_location=device)
        self.load_state_dict(ckpt['model_dict'])
        epoch = ckpt['epoch']
        if self.opt.cluster_num > 0:
            self.data_cluster.load_cluster_txt(epoch, True)
        return epoch

    def print_running_time(self):
        self.time.print_time("encoder_start", "encoder_done")
        self.time.print_time("encoder_done", "cluster_append_data_done")
        self.time.print_time("cluster_append_data_done", "cluster_done")
        self.time.print_time("cluster_done", "latent_done")
        self.time.print_time("latent_done", "decoder_done")

        self.time.clear_all_bufs()

    def print_ckpt_dict(self, dict, mark=""):
        if not "" == mark:
            mark_str = "For " + mark + ". "
        else:
            mark_str = ""
        print(mark_str + "Print dict keys():", dict.keys())
        for key in dict.keys():
            if "dict" in key:
                self.print_ordereddict_for_ckpt_with_mark(key, dict[key])

    def print_ordereddict_for_ckpt_with_mark(self, mark, data):
        data = dict(data)
        print(mark, " data key size", len(data.keys()), " data keys:", data.keys())
        for key in data.keys():
            print("key:", key, "value size:", data[key].size())

    def save_checkpoint_by_layers(self, epoch, opt, filename):
        save_model_content = {}
        if 'x_encoder_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['x_encoder_dict'] = self.x_encoder.state_dict()
        if 'c_encoder_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['c_encoder_dict'] = self.c_encoder.state_dict()
        if 'latent_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['latent_dict'] = self.latent_net.state_dict()
        if 'decoder_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['decoder_dict'] = self.decoder.state_dict()
        if 'cvae_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['cvae_dict'] = self.cvae.state_dict()
        if 'gmm_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['gmm_dict'] = self.gmm_net.state_dict()
        if 'generator_dict' in self.layers_map_for_model[opt.vae_type]:
            save_model_content['generator_dict'] = self.generator.state_dict()
        self.print_ckpt_dict(save_model_content)
        save_model_content['opt'] = opt
        save_model_content['epoch'] = epoch
        torch.save(save_model_content, filename)
        '''torch.save({'x_encoder_dict': self.x_encoder.state_dict(),
                    'c_encoder_dict': self.c_encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'cvae_dict': self.cvae.state_dict(),
                    'gmm_dict': self.gmm_net.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)'''

    def load_checkpoint_by_layers(self, filename):   
        print('Loding model from %s...'%(filename))
        if self.opt.use_cuda:
            ckpt = torch.load(filename, map_location=torch.device(self.opt.gpuid))
        else:
            ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
        self.print_ckpt_dict(ckpt, "model from file")
        loaded_layers_list = []
        if 'x_encoder_dict' in self.layers_map_for_model[self.opt.vae_type] and 'x_encoder_dict' in ckpt:
            self.x_encoder.load_state_dict(ckpt['x_encoder_dict'])
            loaded_layers_list.append("x_encoder_dict")
        if 'c_encoder_dict' in self.layers_map_for_model[self.opt.vae_type] and 'c_encoder_dict' in ckpt:
            self.c_encoder.load_state_dict(ckpt['c_encoder_dict'])
            loaded_layers_list.append("c_encoder_dict")
        if 'latent_dict' in self.layers_map_for_model[self.opt.vae_type] and 'latent_dict' in ckpt:
            self.latent_net.load_state_dict(ckpt['latent_dict']) 
            loaded_layers_list.append("latent_dict")
        if 'decoder_dict' in self.layers_map_for_model[self.opt.vae_type] and 'decoder_dict' in ckpt:
            self.decoder.load_state_dict(ckpt['decoder_dict'])
            loaded_layers_list.append("decoder_dict")
        if 'cvae_dict' in self.layers_map_for_model[self.opt.vae_type] and 'cvae_dict' in ckpt:
            self.cvae.load_state_dict(ckpt['cvae_dict'])
            loaded_layers_list.append("cvae_dict")
        if 'gmm_dict' in self.layers_map_for_model[self.opt.vae_type] and 'gmm_dict' in ckpt:
            self.gmm_net.load_state_dict(ckpt['gmm_dict'])
            loaded_layers_list.append("gmm_dict")
        if 'generator_dict' in self.layers_map_for_model[self.opt.vae_type] and 'generator_dict' in ckpt:
            self.generator.load_state_dict(ckpt['generator_dict'])
            loaded_layers_list.append("generator_dict")
        if self.opt.debug_mode >= 3:
            print("loaded dict:", " ".join(loaded_layers_list))
        epoch = ckpt['epoch']
        if self.opt.cluster_num > 0:
            self.data_cluster.load_cluster_txt(epoch, True)
        #self.opt = ckpt['opt'] # Tmp TODO !!!
        return epoch
