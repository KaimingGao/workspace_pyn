
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from d_vae.encoder import Encoder
from d_vae.decoder import Decoder
from d_vae.dataset import get_dataloader
from d_vae.utils import unmap_pixels



class DiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()
        #
        # [1, 3, 256, 256] -> [1, 8192, 32, 32]
        #
        self.encoder = Encoder()
        #
        # [1, 8192, 32, 32] -> [1, 3, 256, 256]
        #
        self.decoder = Decoder()


    #
    # load dVAE model.
    #
    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=False), strict=True)


    #
    # load encoder and decoder state dict respectively.
    #
    def load_model(self, encoder_file, decoder_file):
        self.encoder.load_state_dict(torch.load(encoder_file, map_location='cpu', weights_only=False), strict=True)
        self.decoder.load_state_dict(torch.load(decoder_file, map_location='cpu', weights_only=False), strict=True)


    #
    # z = [batch, 32, 32]
    #
    @torch.no_grad()
    def decode(self, z):
        z = F.one_hot(z, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()


    #
    # [batch, 3, 256, 256] -> [batch, 8192, 32, 32]
    #
    @torch.no_grad()
    def tokenize(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1).flatten(1)


    #
    # 2**3 = 8
    #
    def pooled_size(self):
        return encoder.pooled_size()


    def vocab_size(self):
        return encoder.vocab_size()


    #
    #
    #
    def forward(self, images, temperature=0.9):
        #
        # logits over vocabulary size, [b, 3, 256, 256] -> [b, 8192, 32, 32]
        #
        z_logits = self.encoder(images)
        #
        # sampling, [b, 8192, 32, 32] -> [b, 8192, 32, 32], one hot
        #
        z = F.gumbel_softmax(z_logits, tau=temperature, dim=1, hard=True)

        ############### Reconstruction Loss ###############
        #
        # generated image, [b, 8192, 32, 32] -> [b, 6, 256, 256]
        #
        x_stats = self.decoder(z).float()
        #
        # [b, 6, 256, 256] -> [b, 3, 256, 256]
        #
        x_stats = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        #
        # reconstruction loss, mse or laplas.
        #
        recon_loss = F.smooth_l1_loss(images, x_stats)

        ##################### KL Loss #####################
        #
        # kl divergence loss, [b, 8192, 32, 32] -> [b, 1024, 8192]
        #
        logits = rearrange(z_logits, 'b n h w -> b (h w) n')
        #
        # [b, 1024, 8192] -> [b, 1024, 8192]
        #
        qy = F.softmax(logits, dim=-1)
        #
        # [b, 1024, 8192]
        #
        log_qy = torch.log(qy + 1e-8)
        #
        # 1.0 / 8192
        #
        log_uniform = torch.log(torch.tensor([1. / self.encoder.vocab_size], device=images.device))
        #
        # kl loss, constrain the variance.
        #
        kl_loss = F.kl_div(log_uniform, log_qy, log_target=True)

        return x_stats, recon_loss, kl_loss

