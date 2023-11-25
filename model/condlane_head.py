import torch
from torch import nn
import numpy as np
import math
import random
import torch.nn.functional as F
from transformer import ConvModule
from loss import CondLaneLoss
from lane import Lane
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class CTnetHead(nn.Module):
    def __init__(self, heads, channels_in, final_kernel=1, head_conv=256):
        super(CTnetHead, self).__init__()
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels_in, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                )
                if "hm" in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels_in, classes, kernel_size=final_kernel, stride=1, padding=final_kernel//2, bias=True)
                if "hm" in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
    
    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z
    def init_weights(self):
        # ctnet_head will init weights during building
        pass

def parse_dynamic_params(params, channels, weight_nums, bias_nums, out_channels=1, mask=True):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    
    #params: (num_ins, n_param)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]
    if mask:
        bias_splits[-1] = bias_splits[-1] - 2.19
    
    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)

        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * out_channels)

    return weight_splits, bias_splits

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step = stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


            
class DynamicMaskHead(nn.Module):
    def __init__(self, num_layers, channels, in_channels, mask_out_stride, weight_nums,
                 bias_nums, disable_coords=False, out_channels=1, 
                 compute_locations_pre=True, location_configs=None):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.in_channels = in_channels
        self.mask_out_stride = mask_out_stride
        self.disable_coords = disable_coords
        self.weight_nums = weight_nums
        self.bias_num = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.out_channels = out_channels
        self.compute_locations_pre = compute_locations_pre
        self.location_configs = location_configs

        if compute_locations_pre and location_configs is not None:
            N, _, H, W = location_configs["size"]
            device = location_configs["device"]
            # shape (W * H, 2)
            locations = compute_locations(H, W, stride=1, device="cpu")
            
            # shape (1, 2, H, W)
            locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, H, W)

            locations[:0, :, :] /= H
            locations[:1, :, :] /= W

            #shape (N, 2, H, W)
            locations = locations.repeat(N, 1, 1, 1)
            self.locations = locations.to(device)
    def forward(self, x, mask_head_params, num_ins, idx=0, is_mask=True):
        N, _, H, W = x.size()
        if not self.disable_coords:
            if self.compute_locations_pre and self.location_configs is not None:
                if self.locations.shape[0] != N: 
                    locations = self.locations.to(x.device)[idx].unsqueeze(0)
                else:
                    locations = self.locations.to(x.device)
            else:
                locations = compute_locations(x.size(2), x.size(3), stride=1, device="cpu")
                locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, H, W)
                locations[:0, :, :] /= H
                locations[:1, :, :] /= W
                locations = locations.repeat(N, 1, 1, 1)
                locations = locations.to(x.device)

            # shape N, 5, H, W

            x = torch.cat([locations, x], dim=1)

        mask_head_inputs = []
        for idx in range(N):
            # reshape each item in batch (1, num_instance, H, W)
            mask_head_inputs.append(x[idx:idx + 1, ...].repeat(1, num_ins[idx], 1, 1))
        # shape (N, total_num_ins, H, W)
        mask_head_inputs = torch.cat(mask_head_inputs, 1)
        num_insts = sum(num_ins)
        #shape (1, N * num_insts, H, W)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = parse_dynamic_params(mask_head_params, self.channels, self.weight_nums, self.bias_num, out_channels=self.out_channels, mask=is_mask)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, num_insts)
        #shape (1, out_channels, H, W)
        mask_logits = mask_logits.view(1, -1, H, W)
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, weight=w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x     
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CondLaneHead(nn.Module):

    def __init__(self, heads, in_channels, num_classes, mask_size, head_channels=64, head_layers=1,
                 disable_coords=False, branch_in_channels=288, branch_channels=64,
                 branch_out_channels=64, reg_branch_channels=32, branch_num_conv=1,
                 norm_type= "BN", hm_idx=-1, mask_idx=0, compute_locations_pre=True,
                 location_configs=None, mask_norm_act=True, regression=True):
        super(CondLaneHead, self).__init__()
        self.mask_size = mask_size
        self.num_classes=num_classes
        self.hm_idx = hm_idx
        self.mask_idx = mask_idx
        self.regression = regression
        if mask_norm_act:
            final_norm_type = "BN"
            final_act_type = "ReLU"
        else:
            final_act_type = None
            final_act_type = None
        mask_branch = []
        mask_branch.append(ConvModule(sum(in_channels), branch_channels, kernel_size=3, padding=1, norm=norm_type))
        for i in range(branch_num_conv):
            mask_branch.append(ConvModule(branch_channels, branch_channels, kernel_size=3, padding=1, norm=norm_type))
        mask_branch.append(ConvModule(branch_channels, branch_out_channels, kernel_size=3, padding=1, norm=final_norm_type, act=final_act_type))
        self.add_module("mask_branch", nn.Sequential(*mask_branch))

        self.mask_weight_nums , self.mask_bias_nums = self.cal_num_params(head_layers, disable_coords, head_channels, out_channels=1)
        self.num_mask_params = sum(self.mask_weight_nums) + sum(self.mask_bias_nums)

        self.reg_weight_nums, self.reg_bias_nums = self.cal_num_params(head_layers, disable_coords, head_channels, out_channels=1)
        self.num_reg_params = sum(self.reg_weight_nums) + sum(self.reg_bias_nums)

        if self.regression:
            self.num_gen_params = self.num_mask_params + self.num_reg_params
        else:
            self.num_gen_params = self.num_mask_params
            self.num_reg_params = 0

        self.mask_head = DynamicMaskHead(head_layers, branch_out_channels, branch_out_channels, 1, self.mask_weight_nums, self.mask_bias_nums,
                                         disable_coords=False, compute_locations_pre=compute_locations_pre, location_configs=location_configs)
        
        if self.regression:
            self.reg_head = DynamicMaskHead(head_layers, branch_out_channels, branch_out_channels, 1, self.reg_weight_nums,
                                            self.reg_bias_nums, disable_coords=False, out_channels=1,
                                            compute_locations_pre=compute_locations_pre, location_configs=location_configs)
        if "param" not in heads:
            heads["params"] = num_classes * (self.num_mask_params + self.num_reg_params)
        self.ctnet_head = CTnetHead(heads, channels_in=branch_in_channels, final_kernel=1, head_conv=branch_in_channels)
        self.feat_width = location_configs["size"][-1]
        self.mlp = MLP(self.feat_width, 64, 2, 2)


        loss_weight_dict = dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    )
        self.loss_impl = CondLaneLoss(weights=loss_weight_dict)

    def loss(self, output, batch):
        batch.pop("meta")
        return self.loss_impl(output, **batch)
    def cal_num_params(self, num_layers, disable_coords, channels, out_channels=1):
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    weight_nums.append((channels + 2) * out_channels)
                else:
                    weight_nums.append(channels * out_channels)
                bias_nums.append(out_channels)

            elif l ==0:
                if not disable_coords:
                    weight_nums.append((channels + 2) * channels)
                else:
                    weight_nums.append(channels * channels)
                bias_nums.append(channels)
            
            else:
                weight_nums.append(channels * channels)
                bias_nums.append(channels)
        return weight_nums, bias_nums
    
    def parse_gt(self, gts, device):
        reg = (torch.from_numpy(gts['reg']).to(device)).unsqueeze(0)

        reg_mask = (torch.from_numpy(gts['reg_mask']).to(device)).unsqueeze(0)
        row = (torch.from_numpy(
            gts['row']).to(device)).unsqueeze(0).unsqueeze(0)
        row_mask = (torch.from_numpy(
            gts['row_mask']).to(device)).unsqueeze(0).unsqueeze(0)
        if 'range' in gts:
            lane_range = torch.from_numpy(gts['range']).to(device) # new add: squeeze 
            #lane_range = (gts['range']).to(device).squeeze(0) # new add: squeeze 
        else:
            lane_range = torch.zeros((1, reg_mask.shape[-2]),
                                     dtype=torch.int64).to(device)
        return reg, reg_mask, row, row_mask, lane_range
    def parse_pos(self, gt_masks, hm_shape, device, mask_shape=None):
        b = len(gt_masks)
        n = self.num_classes
        hm_h, hm_w = hm_shape[:2]
        if mask_shape is None:
            mask_h, mask_w = hm_shape[:2]
        else:
            mask_h, mask_w = mask_shape[:2]
        poses = []
        regs = []
        reg_masks = []
        rows = []
        row_masks = []
        lane_ranges = []
        labels = []
        num_ins = []
        
        for idx, m_img in enumerate(gt_masks):
            num = 0
            for m in m_img:
                gts = self.parse_gt(m, device=device)
                reg, reg_mask, row, row_mask, lane_range = gts
                label = m["label"]
                num += len(m["points"])
                for p in m["points"]:
                    pos = idx * n * hm_h * hm_w + label * hm_h * hm_w + p[1] * hm_w + p[0]
                    poses.append(pos)
                for i in range(len(m["points"])):
                    labels.append(label)
                    regs.append(reg)
                    reg_masks.append(reg_mask)
                    rows.append(row)
                    row_masks.append(row_mask)
                    lane_ranges.append(lane_range)
            if num == 0:
                reg = torch.zeros((1, 1, mask_h, mask_w)).to(device)
                reg_mask = torch.zeros((1, 1, mask_h, mask_w)).to(device)
                row = torch.zeros((1, 1, mask_h)).to(device)
                row_mask = torch.zeros((1, 1, mask_h)).to(device)
                lane_range = torch.zeros((1, mask_h), dtype=torch.int64).to(device)
                label = 0
                pos = idx * n * hm_h * hm_w + random.randint(0, n * hm_h * hm_w - 1)
                num = 1
                labels.append(label)
                poses.append(pos)
                regs.append(reg)
                reg_masks.append(reg_mask)
                rows.append(row)
                row_masks.append(row_mask)
                lane_ranges.append(lane_range)

            num_ins.append(num)
        if len(reg) > 0:
            regs = torch.cat(regs, 1)
            reg_masks = torch.cat(reg_masks, 1)
            rows = torch.cat(rows, 1)
            row_masks = torch.cat(row_masks, 1)
            lane_ranges = torch.cat(lane_ranges, 0)
        
        gts = dict(
            gt_reg=regs,
            gt_reg_mask=reg_masks,
            gt_rows=rows,
            gt_row_masks=row_masks,
            gt_ranges=lane_ranges)
        
        return poses, labels, num_ins, gts

    def forward_train(self, output, batch):
        img_metas = batch['img_metas']._data[0]
        gt_batch_masks = [m["gt_masks"] for m in img_metas]
        hm_shape = img_metas[0]["hm_shape"]
        mask_shape = img_metas[0]["mask_shape"]
        inputs = output
        pos, label, num_ins, gts = self.parse_pos(gt_batch_masks, hm_shape, inputs[0].device, mask_shape)
        batch.update(gts)
        x_list = list(inputs)

        f_hm = x_list[self.hm_idx]
        # print(f_hm.shape)
        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]

        #f_mask
        z = self.ctnet_head(f_hm)
        hm, params = z["hm"], z["params"]

        h_hm, w_hm = hm.size()[2:]
        h_mask, w_mask = f_mask.size()[2:]
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)

        mask_branch = self.mask_branch(f_mask)

        reg_branch = mask_branch

        params = params.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.num_gen_params)
        pos_tensor = torch.from_numpy(np.array(pos, dtype=np.float64)).long().to(params.device).unsqueeze(1)

        pos_tensor = pos_tensor.expand(-1, self.num_gen_params)
        mask_pos_tensor = pos_tensor[:, :self.num_mask_params]
        reg_pos_tensor = pos_tensor[:, self.num_mask_params:]

        if pos_tensor.size()[0] == 0:
            masks = None
            feat_range = None
        else:
            mask_params = params[:, :self.num_mask_params].gather(0, mask_pos_tensor)
            masks = self.mask_head(mask_branch, mask_params, num_ins)
            if self.regression:
                reg_params = params[:, self.num_mask_params:].gather(0, reg_pos_tensor)
                regs = self.reg_head(reg_branch, reg_params, num_ins)
            else:
                regs = masks

            feat_range = masks.permute(0, 1, 3, 2).view(sum(num_ins), w_mask, h_mask)

            feat_range = self.mlp(feat_range)
        
        batch.update(dict(mask_branch=mask_branch, reg_branch=reg_branch))
        return hm, regs, masks, feat_range, [mask_branch, reg_branch]
    
    def ctdet_decode(self, heat, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        def _format(heat, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,
                    'id_class': id_class,
                    'score': score
                })
            return ret

        heat_nms = _nms(heat)
        # print(heat.shape, heat_nms.shape)
        heat_nms = heat_nms.permute(1, 2, 0).detach().cpu().numpy()
        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        # heat_nms = heat_nms.permute(0, 2, 3, 1).detach().cpu().numpy()
        # inds = np.where(heat_nms > thr)
        # print(len(inds))
        # seeds = _format(heat_nms, inds)
        return seeds
    
    def forward_test(self, inputs, hack_seeds=None, hm_thr=0.5):
        def parse_pos(seeds, batchsize, num_classes, h, w, device):
            pos_list = [[p["coord"], p["id_class"] - 1] for p in seeds]
            poses = []
            for p in pos_list:
                [c, r], label = p
                pos = label * h * w + r * w + c
                poses.append(pos)
            poses = torch.from_numpy(np.array(poses, np.longlong)).long().to(device).unsqueeze(1)
            return poses
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]
        z = self.ctnet_head(f_hm)
        h_hm, w_hm = f_hm.size()[2:]
        h_mask, w_mask = f_mask.size()[2:]
        hms, params = z["hm"], z["params"]
        hms = torch.clamp(hms.sigmoid(), min=1e-4, max=1 -1e-4)
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)
        mask_branchs = self.mask_branch(f_mask)
        reg_branchs = mask_branchs
        params = params.permute(0, 1, 3, 4, 2).contiguous().view(m_batchsize, -1, self.num_gen_params)
        batch_size, num_classes, h, w = hms.size()
        out_seeds, out_hm = [], []
        idx= 0
        for hm, param, mask_branch, reg_branch in zip(hms, params, mask_branchs, reg_branchs):
            mask_branch = mask_branch.unsqueeze(0)
            reg_branch = reg_branch.unsqueeze(0)
            seeds = self.ctdet_decode(hm, thr=hm_thr)
            if hack_seeds is not None:
                seeds = hack_seeds
            pos_tensor = parse_pos(seeds, batch_size, num_classes, h, w, hm.device)
            pos_tensor = pos_tensor.expand(-1, self.num_gen_params)
            num_ins = [pos_tensor.size()[0]]
            mask_pos_tensor = pos_tensor[:, :self.num_mask_params]
            if self.regression:
                reg_pos_tensor = pos_tensor[:, self.num_mask_params:]
            if pos_tensor.size()[0] == 0:
                seeds = []
            else:
                mask_params = param[:, :self.num_mask_params].gather(0, mask_pos_tensor)
                masks = self.mask_head(mask_branch, mask_params, num_ins, idx)

                if self.regression:
                    reg_params = param[:, self.num_mask_params:].gather(0, reg_pos_tensor) 
                    regs = self.reg_head(reg_branch, reg_params, num_ins, idx)
                else:
                    regs = masks
                feat_range = masks.permute(0, 1, 3, 2).view(sum(num_ins), w_mask, h_mask)
                feat_range = self.mlp(feat_range)
                for i in range(len(seeds)):
                    seeds[i]["reg"] = regs[0, i:i + 1, :, :]
                    m = masks[0, i:i+1, :, :]
                    seeds[i]["mask"] = m
                    seeds[i]["range"] = feat_range[i:i+1]
            out_seeds.append(seeds)
            out_hm.append(hm)
            idx ==1
        output = {"seeds" : out_seeds, "hm": out_hm}
        return output
    def forward(
            self,
            x_list,
            **kwargs):
        if self.training:
            return self.forward_train(x_list, kwargs['batch'])
        return self.forward_test(x_list, )
    

    def init_weights(self):
        # ctnet_head will init weights during building
        pass