
import torch
import torch.nn.functional as F

from torch import nn
from transformers.models.bert.modeling_bert import BertModel



#
#
#
class TDVAEModel(nn.Module):
    def __init__(
        self,
        dim=768,
        bert_conf=None,
        bert_file=None,
        dropout=0.1,
        codes_size=512,
        query_leng=16,
        vocab_size=21128,
    ):
        super().__init__()

        self.codes_size = codes_size
        self.vocab_size = vocab_size
        #
        # transformer encoder
        #
        self.encoder = BertModel(config=bert_conf) if bert_conf is not None and bert_file is None else BertModel.from_pretrained(bert_file, ignore_mismatched_sizes=True)
        #
        # quantizer: sampling layer.
        #
        self.sampling = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, codes_size), nn.GELU(), nn.Dropout(dropout))
        #
        # quantizer: codebook layer.
        #
        self.codebook = nn.Sequential(nn.Linear(codes_size, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim))
        #
        # output layer.
        #
        self.output = nn.Sequential(nn.Linear(dim, self.vocab_size), nn.Dropout(dropout), nn.LayerNorm(self.vocab_size))


    #
    #
    #
    def load(self, model_file=None, strict=True):
        self.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=False), strict=strict)



    #
    #
    #
    def forward(self, query=None, label=None, mask=None, tau=0.9, codes_leng=3):

        ###################### Encoder ####################

        x = self.encoder(query, mask).last_hidden_state
        #
        # [cls]
        #
        x = x[:, 0, :]

        ###################### Latent #####################

        x = self.sampling(x)
        #
        # P(z|x)
        #
        z = F.gumbel_softmax(x, dim=-1, tau=tau, hard=False)

        weight, indice = torch.topk(z, k=codes_leng, largest=True)

        z = F.one_hot(indice, num_classes=self.codes_size).float()

        z = self.codebook(z)

        ###################### KL Loss ####################

        log_p = F.log_softmax(x, dim=-1)

        log_uniform = torch.log(torch.tensor([1. / self.codes_size], device=query.device))

        kl_loss = F.kl_div(log_uniform, log_p, log_target=True)

        ###################### LM LOSS ####################
        #
        # P(y|z) * P(z|x)
        #
        y = self.output(z) * weight.unsqueeze(-1)
        #
        # SUM( P(y|z) * P(z|x) )
        #
        y = torch.sum(y, dim=1)
        #
        # P(y|x) = P(y|z) * P(z|x)
        #
        lm_loss = F.cross_entropy(y, label)

        return kl_loss, lm_loss



    @torch.no_grad()
    def generate(self, query=None, mask=None, codes_leng=3, decode=False):

        ###################### Encoder ####################

        x = self.encoder(query, mask).last_hidden_state
        #
        # [cls], [batch, 768]
        #
        x = x[:, 0, :]
        ###################### Latent #####################
        #
        # [batch, codes_size]
        #
        x = self.sampling(x)
        #
        # P(z|x), [batch, codes_size]
        #
        z = torch.softmax(x, dim=-1)
        #
        # [batch, codes_leng]
        #
        weight, indice = torch.topk(z, k=codes_leng, largest=True)
        #
        # early return.
        #
        if decode is False:
            return indice
            # return torch.sort(indice).values

        z = F.one_hot(indice, num_classes=self.codes_size).float()

        z = self.codebook(z)
        #
        # P(y|z) * P(z|x)
        #
        y = self.output(z)
        #
        # SUM( P(y|z) * P(z|x) )
        #
        y = torch.sum(y * weight.unsqueeze(-1), dim=1)

        return indice, torch.topk(y, 20, largest=True).indices




    @torch.no_grad()
    def show(self, indice=[0]):

        z = F.one_hot(torch.tensor(indice), num_classes=self.codes_size).float()

        z = self.codebook(z)

        y = self.output(z)

        return torch.topk(y.softmax(dim=-1), 100, largest=True)


