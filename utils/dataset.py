from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from os.path import join
import numpy as np
import os


class fixation_dataset(Dataset):
    def __init__(self, fixs, img_ftrs_dir, logit_lens_dir=None, logit_lens_top_k=5):
        self.fixs = fixs
        self.img_ftrs_dir = img_ftrs_dir
        self.logit_lens_dir = logit_lens_dir
        self.logit_lens_top_k = logit_lens_top_k
        self.use_logit_lens = logit_lens_dir is not None

        
    def __len__(self):
        return len(self.fixs)
        
    def __getitem__(self, idx):
        fixation = self.fixs[idx]

        image_ftrs = torch.load(join(self.img_ftrs_dir, fixation['task'].replace(' ', '_'), fixation['img_name'].replace('jpg', 'pth'))).unsqueeze(0)

        result = {
            'task': fixation['task'], 
            'tgt_y': fixation['tgt_seq_y'].float(), 
            'tgt_x': fixation['tgt_seq_x'].float(), 
            'tgt_t': fixation['tgt_seq_t'].float(),
            'src_img': image_ftrs
        }
        
        if self.use_logit_lens:
            img_id = fixation['img_name'].replace('.jpg', '')
            logit_lens_path = join(self.logit_lens_dir, 'semantics', f'{img_id}_word_vectors.npy')
            
            if not os.path.exists(logit_lens_path):
                compressed_path = join(self.logit_lens_dir, 'compressed', f'{img_id}_word_vectors.npz')
                if os.path.exists(compressed_path):
                    logit_lens_data = np.load(compressed_path)
                    logit_lens_vectors = logit_lens_data['arr_0'] if 'arr_0' in logit_lens_data else logit_lens_data[list(logit_lens_data.keys())[0]]
                else:
                    logit_lens_vectors = None
            else:
                logit_lens_vectors = np.load(logit_lens_path)
            
            if logit_lens_vectors is not None:
                # keep only first top_k semantic vectors per patch
                if len(logit_lens_vectors.shape) == 3:  # (num_patches, top_k, embedding_dim)
                    logit_lens_vectors = logit_lens_vectors[:, :self.logit_lens_top_k, :]
                result['logit_lens'] = torch.from_numpy(logit_lens_vectors).float()
            else:
                result['logit_lens'] = None
        else:
            result['logit_lens'] = None
        
        return result
        

class COCOSearch18Collator(object):
    def __init__(self, embedding_dict, max_len, im_h, im_w, patch_size, use_logit_lens=False):
        self.embedding_dict = embedding_dict
        self.max_len = max_len
        self.im_h = im_h
        self.im_w = im_w
        self.patch_size = patch_size
        self.PAD = [-3, -3, -3]
        self.use_logit_lens = use_logit_lens

    def __call__(self, batch):
        batch_tgt_y = []
        batch_tgt_x = []
        batch_tgt_t = []
        batch_imgs = []
        batch_tasks = []
        batch_logit_lens = []
        
        for t in batch:
            batch_tgt_y.append(t['tgt_y'])
            batch_tgt_x.append(t['tgt_x'])
            batch_tgt_t.append(t['tgt_t'])
            batch_imgs.append(t['src_img'])
            batch_tasks.append(self.embedding_dict[t['task']])
            if self.use_logit_lens:
                batch_logit_lens.append(t['logit_lens'])
        
        batch_tgt_y.append(torch.zeros(self.max_len))
        batch_tgt_x.append(torch.zeros(self.max_len))
        batch_tgt_t.append(torch.zeros(self.max_len))
        batch_tgt_y = pad_sequence(batch_tgt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_tgt_x = pad_sequence(batch_tgt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_tgt_t = pad_sequence(batch_tgt_t, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        
        batch_imgs = torch.cat(batch_imgs, dim = 0)
        batch_tgt = torch.cat([batch_tgt_y, batch_tgt_x, batch_tgt_t], dim = -1).long().permute(1, 0, 2)
        batch_firstfix = torch.tensor([(self.im_h//2)*self.patch_size, (self.im_w//2)*self.patch_size]).unsqueeze(0).repeat(batch_imgs.size(0), 1)
        batch_tgt_padding_mask = batch_tgt[:, :, 0] == self.PAD[0]
        
        if self.use_logit_lens:
            valid_logit_lens = [ll for ll in batch_logit_lens if ll is not None]
            if len(valid_logit_lens) > 0:
                batch_logit_lens_tensor = torch.stack(valid_logit_lens, dim=0)
                if batch_logit_lens_tensor.size(0) < batch_imgs.size(0):
                    last_vector = batch_logit_lens_tensor[-1:]
                    padding = last_vector.repeat(batch_imgs.size(0) - batch_logit_lens_tensor.size(0), 1, 1, 1)
                    batch_logit_lens_tensor = torch.cat([batch_logit_lens_tensor, padding], dim=0)
            else:
                batch_logit_lens_tensor = None
        else:
            batch_logit_lens_tensor = None
        
        if batch_logit_lens_tensor is not None:
            return batch_imgs, batch_tgt, batch_tgt_padding_mask, torch.tensor(batch_tasks), batch_firstfix, batch_logit_lens_tensor
        else:
            return batch_imgs, batch_tgt, batch_tgt_padding_mask, torch.tensor(batch_tasks), batch_firstfix, None

        
        
