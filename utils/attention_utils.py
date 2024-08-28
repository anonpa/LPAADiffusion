import torch 
from utils.dist import get_gaussian_map, init_loc
from utils.cache_gradient import hidden_state_gradient

class AttendExciteAttnProcessor_custom:
    def __init__(self, attnstore, place_in_unet, controller):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.controller = controller
        self.store_attn_map = False
        self.attn_map = None
        self.ema_attn_map = None
        self.debug_map_real = None
        self.debug_map_prev = None
        self.debug_map_diff = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)

        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # if hidden_states.shape[1] == 256 and is_cross:
        #     self.debug_map_real = attention_probs[8:]

        is_target_layer = self.controller.current_layer in self.controller.target_layers if self.controller.target_layers is not None else True
        is_target_res = hidden_states.shape[1] == self.controller.target_res

        if is_target_res and is_cross:
            self.controller.current_layer += 1

        # print(self.controller.target_res, self.controller.current_layer, self.controller.target_layers)

        if is_cross and is_target_layer and is_target_res:

            scale = self.controller.hyper_config['scale']
            preserve_scale = self.controller.hyper_config['preserve_scale']
            lse_scale = self.controller.hyper_config['lse_scale']
            align_scale = self.controller.hyper_config['align_scale']
            cache = self.controller.hyper_config['cache']
            disalign_scale = self.controller.hyper_config['disalign_scale']
            beta = self.controller.hyper_config['beta']


            if self.controller.current_step % cache == 0:
                self.attn_map = attn.get_attention_scores(query, key, attention_mask)[8:]




            if len(self.controller.aligned) != 0:
                ns, adjs = zip(*self.controller.aligned)
            else:
                ns = self.controller.ns 
                adjs = list()
            

            grad = hidden_state_gradient(self.attn_map, ns, adjs, key[8:], attn.scale, 
                                         attn.to_q.weight, 
                                         self.controller.preserve_prior.to(self.attn_map.device), 
                                         preserve_scale=preserve_scale, 
                                         lse_scale=lse_scale, 
                                         align_scale=align_scale, 
                                         disalign_scale=disalign_scale)
            q_scale = scale * (beta ** self.controller.current_step)
            hidden_states = hidden_states -  q_scale * grad 


        

                        



        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        

        return hidden_states






class CustomController():

    def __init__(self, aligned, ns, max_step, hyper_config):

        self.aligned = aligned
        self.ns = ns
        self.max_step = max_step
        self.current_step = -1
        self.hyper_config = hyper_config
        preserve_prior = self._get_preserve_prior()
        self.preserve_prior = preserve_prior
        self.current_layer = 0
        self.target_layers = hyper_config['target_layers']
        self.target_res = hyper_config['target_res']
    

    def _get_preserve_prior(self, eps=1e-5):
        if len(self.aligned) != 0:
            ns, adjs = zip(*self.aligned)
        else:
            ns = self.ns 
            adjs = list() 

        locs = init_loc(len(ns))
        priors = list()
        for loc in locs:
            priors.append(get_gaussian_map(loc, normalize=True))

        priors = torch.stack(priors, dim=0) + eps
        return priors
        





