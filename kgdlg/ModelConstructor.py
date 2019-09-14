from kgdlg.modules.Encoder import EncoderRNN
from kgdlg.modules.Decoder import AttnDecoder,InputFeedDecoder,MHAttnDecoder
from kgdlg.modules.Embedding import Embedding
from kgdlg.Model import NMTModel,MoSGenerator,CVAE,CvaeDialog,LatentNet,VariationalInference,DataCluster
import torch
import torch.nn as nn
import kgdlg.IO as IO




def create_emb_for_encoder_and_decoder(src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       padding_idx):

    embedding_encoder = Embedding(src_vocab_size,src_embed_size,padding_idx)
    embedding_decoder = Embedding(tgt_vocab_size,tgt_embed_size,padding_idx)

        
    return embedding_encoder, embedding_decoder


def create_encoder(opt,embedding):
    
    rnn_type = opt.rnn_type
    input_size = opt.embedding_size
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    dropout = opt.dropout
    bidirectional = opt.bidirectional

    encoder = EncoderRNN(rnn_type,
                        embedding,
                        input_size,
                        hidden_size,
                        num_layers,
                        dropout,
                        bidirectional)

    return encoder

def create_decoder(opt,embedding):

    decoder_type = opt.decoder_type
    rnn_type = opt.rnn_type  
    atten_model = opt.atten_model
    input_size = opt.embedding_size
    if (opt.src_tgt_latent_merge_type in [0]) and (opt.vae_type in [6, 7, 8]):
        hidden_size = opt.hidden_size * 2
    elif (opt.use_src_or_tgt_attention in [3]) and (opt.vae_type in [6, 7, 8]):
        hidden_size = opt.hidden_size * 2
    else:
        hidden_size = opt.hidden_size

    num_layers = opt.num_layers
    dropout = opt.dropout 

    if decoder_type == 'AttnDecoder':
        decoder = AttnDecoder(rnn_type,
                                embedding,
                                atten_model,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout)
    elif decoder_type == 'InputFeedDecoder':
        decoder = InputFeedDecoder(rnn_type,
                                atten_model,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout)    

    elif decoder_type == 'MHAttnDecoder':
        decoder = MHAttnDecoder(rnn_type,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout)                                     
   

    return decoder

def create_generator(input_size, output_size):
    # generator = MoSGenerator(5, input_size, output_size)
    generator = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LogSoftmax(dim=-1))
    return generator

def create_cvae(x_feat_size,c_feat_size,latent_size):
    net = CVAE(x_feat_size,c_feat_size,latent_size)
    return net

def create_gmm(opt):
    net = VariationalInference(opt)
    return net

def create_latent_net(opt):
    net = LatentNet(opt)
    return net

def create_data_cluster(opt):
    net = DataCluster(opt)
    return net

def create_base_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    opt.variable_src_dict = fields["src"].vocab
    opt.variable_tgt_dict = fields["tgt"].vocab
    padding_idx = fields['src'].vocab.stoi[IO.PAD_WORD]
    #enc_embedding, dec_embedding = \
    #        create_emb_for_encoder_and_decoder(src_vocab_size,
    #                                            tgt_vocab_size,
    #                                            opt.embedding_size,
    #                                            opt.embedding_size,
    #                                            padding_idx)

    # create embedding and encoder
    enc_embedding = Embedding(src_vocab_size,opt.embedding_size,padding_idx)
    dec_embedding = Embedding(tgt_vocab_size,opt.embedding_size,padding_idx)
    #src_enc_embedding = None
    if opt.vae_type in [6, 7, 8] and 1 == opt.embedding_type:
        src_enc_embedding = Embedding(src_vocab_size,opt.embedding_size,padding_idx)
        c_encoder = create_encoder(opt,src_enc_embedding)
    else:
        c_encoder = create_encoder(opt,enc_embedding)

    if opt.embedding_type in [1, 2]:
        x_encoder = create_encoder(opt,enc_embedding)
    elif opt.embedding_type in [3]:
        x_encoder = create_encoder(opt,dec_embedding)
        
    latent_net = create_latent_net(opt)
    decoder = create_decoder(opt,dec_embedding)
    cvae_net = create_cvae(opt.hidden_size,opt.hidden_size,opt.latent_size)
    gmm_net = create_gmm(opt)
    #if opt.vae_type == 3 or opt.vae_type == 4:
    #    gmm_net = create_gmm(opt)
    #else:
    #    gmm_net = None
    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    data_cluster = create_data_cluster(opt)
    # model = NMTModel(enc_embedding, 
    #                  dec_embedding, 
    #                  encoder, 
    #                  decoder, 
    #                  generator)

    # model.apply(weights_init)
    model = CvaeDialog(x_encoder,c_encoder,decoder, cvae_net, gmm_net, latent_net, generator, data_cluster, opt)
    return model

