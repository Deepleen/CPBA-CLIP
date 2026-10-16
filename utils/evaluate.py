import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score
from scipy.ndimage import gaussian_filter
from utils.tools import normalize
from tqdm import tqdm

import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score


def evaluate_pre(test_dataloader, model, linearlayer, New_Lan_Embed,prompt_ceshi, device, args, obj_list):
    model.eval()
    linearlayer.eval()
    New_Lan_Embed.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    id = 0
    for items in test_dataloader:
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        use_attr_text = getattr(args, 'use_attribute_text', True)
        if isinstance(use_attr_text, str):
            use_attr_text = use_attr_text.lower() in ('true', '1', 'yes')
        txt_content = items.get('txt_content', None) if use_attr_text else None
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features, patch_tokens = model.encode_image(image, args.features_list)

            # 1. 生成广义文本特征（不带属性）
            class_token_global = New_Lan_Embed.before_extract_feat(
                patch_tokens, image_features, use_global=args.use_global
            )
            text_embeddings_global = prompt_ceshi.forward_ensemble(
                model, class_token_global, device, txt_content=None
            )
            text_embeddings_global = text_embeddings_global.permute(0, 2, 1)

            # sample
            patch_tokens = linearlayer(patch_tokens)
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
               
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (New_Lan_Embed.prompt_temp.exp() * patch_tokens[layer] @ text_embeddings_global)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            
            anomaly_map = np.sum(anomaly_maps, axis=0)[0]
            results['anomaly_maps'].append(anomaly_map)
    # metrics
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_px_ls.append(ap_px)

    ap_mean = np.mean(ap_px_ls)
    model.train()
    linearlayer.train()
    New_Lan_Embed.train()
    del results, gt_px, pr_px
    return ap_mean 

def evaluate_post(test_dataloader, model, linearlayer, New_Lan_Embed,Zero_try, prompt_ceshi, device, args, obj_list):
    """Return ``(ap_px, ap_regional, ap_global)`` pixel AP values.

    ``ap_global`` is the CPG/global branch AP, ``ap_regional`` is the
    BAN/post branch AP, and ``ap_px`` is the test-time CPG/BAN fused AP used
    for checkpoint comparison.
    """
    model.eval()
    linearlayer.eval()
    New_Lan_Embed.eval()
    Zero_try.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_map_raw'] = []
    results['anomaly_map_new'] = []
    for items in test_dataloader:
        image = items['img'].to(device)
        cls_name = items['cls_name']
        use_attr_text = getattr(args, 'use_attribute_text', True)
        if isinstance(use_attr_text, str):
            use_attr_text = use_attr_text.lower() in ('true', '1', 'yes')
        txt_content = items.get('txt_content', None) if use_attr_text else None
        depth = items.get('depth', None)
        if depth is not None:
            depth = depth.to(device)
        results['cls_names'].extend(cls_name)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask.squeeze(1).numpy())  # px
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features, patch_tokens = model.encode_image(image, args.features_list)
            use_depth = getattr(args, 'use_depth', False)
            use_regional = getattr(args, 'use_regional', True)
            if use_regional and use_depth and depth is not None:
                _, patch_tokens_depth = model.encode_depth(depth, args.features_list)
            else:
                patch_tokens_depth = None

            # Keep this raw branch identical to test_V3.py: CPG/global prompt
            # with the short text template.
            class_token_global = New_Lan_Embed.before_extract_feat(
                patch_tokens, image_features, use_global=args.use_global
            )
            text_embeddings_global = prompt_ceshi.forward_ensemble(
                model, class_token_global, device, txt_content=None
            )
            text_embeddings_global = text_embeddings_global.permute(0, 2, 1)

            # Keep the BAN branch identical to test_V3.py: base prompt with
            # attribute text, without global visual context.
            class_token_regional = New_Lan_Embed.before_extract_feat(
                patch_tokens, image_features, use_global=False
            )
            text_embeddings_regional = prompt_ceshi.forward_ensemble(
                model, class_token_regional, device, txt_content=txt_content
            )
            text_embeddings_regional = text_embeddings_regional.permute(0, 2, 1)

            anomaly_maps_new = []
            for layer in range(len(patch_tokens)):
                dense_feature = patch_tokens[layer][:, 1:, :].clone()
                dense_feature = dense_feature / dense_feature.norm(dim=-1, keepdim=True)
                if not use_regional:
                    anomaly_map = (New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings_regional)
                else:
                    if patch_tokens_depth is not None:
                        dense_depth = patch_tokens_depth[layer][:,1:,:].clone()
                        dense_depth = dense_depth / dense_depth.norm(dim=-1, keepdim=True)
                        F_t_a, _ = Zero_try(text_embeddings_regional.permute(0,2,1), dense_feature, F_s_depth=dense_depth)
                    else:
                        F_t_a, _ = Zero_try(text_embeddings_regional.permute(0,2,1), dense_feature, F_s_depth=None)
                    anomaly_map = (Zero_try.prompt_temp_l1.exp() * dense_feature @ F_t_a.permute(0,2,1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps_new.append(anomaly_map.cpu().numpy())
            
            patch_tokens_linear = linearlayer(patch_tokens)
            anomaly_maps_raw = []
            for layer in range(len(patch_tokens_linear)):
                dense_feature = patch_tokens_linear[layer].clone()
                dense_feature = dense_feature / dense_feature.norm(dim=-1, keepdim=True)
                anomaly_map = (
                    New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings_global
                )
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps_raw.append(anomaly_map.cpu().numpy())

            # test_V3.py averages feature layers before pixel-map fusion.
            results['anomaly_map_raw'].append(np.mean(anomaly_maps_raw, axis=0))
            results['anomaly_map_new'].append(np.mean(anomaly_maps_new, axis=0))

    # Match test_V3.py exactly: category-wise raw/post fusion, Gaussian
    # smoothing over the spatial axes, then min-max normalization before AP.
    ap_px_ls = []
    ap_regional_ls = []
    ap_global_ls = []
    names_list = np.array(results['cls_names'])
    anomaly_maps_raw = np.concatenate(results['anomaly_map_raw'], axis=0)
    anomaly_maps_new = np.concatenate(results['anomaly_map_new'], axis=0)
    gts_pixel = np.concatenate(results['imgs_masks'], axis=0)
    alpha = 0.2
    for obj in obj_list:
        object_index = np.where(names_list == obj)[0]
        pr_px_1 = anomaly_maps_raw[object_index, :, :].copy()
        pr_px_2 = anomaly_maps_new[object_index, :, :].copy()
        gt_px = gts_pixel[object_index, :, :].copy()
        ap_global = average_precision_score(gt_px.ravel(), pr_px_1.ravel())
        ap_global_ls.append(ap_global)
        ap_regional = average_precision_score(gt_px.ravel(), pr_px_2.ravel())
        ap_regional_ls.append(ap_regional)
        pr_px = normalize(
            gaussian_filter(alpha * pr_px_1 + (1 - alpha) * pr_px_2, sigma=8, axes=(1, 2))
        )
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_px_ls.append(ap_px)
    ap_px_mean = np.mean(ap_px_ls)
    ap_regional_mean = np.mean(ap_regional_ls)
    ap_global_mean = np.mean(ap_global_ls)
    model.train()
    linearlayer.train()
    New_Lan_Embed.train()
    Zero_try.train()
    del results, gt_px, pr_px, pr_px_1, pr_px_2
    return ap_px_mean, ap_regional_mean, ap_global_mean




def evaluate(test_dataloader, model, linearlayer, New_Lan_Embed,Zero_try, prompt_ceshi, device, args, obj_list):
    model.eval()
    linearlayer.eval()
    New_Lan_Embed.eval()
    Zero_try.eval()

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['pr_sp'] = []
    results['gt_sp'] = []
    results['anomaly_map_raw'] = []
    results['anomaly_map_new'] = []
    results['path'] = []
    id = 0
    test_dataloader = tqdm(test_dataloader)
    print("---------------------------start evaluation---------------------------")
    for items in test_dataloader:
        id = id + 1
        #print(id)
        image = items['img'].to(device)
        cls_name = items['cls_name']
        use_attr_text = getattr(args, 'use_attribute_text', True)
        if isinstance(use_attr_text, str):
            use_attr_text = use_attr_text.lower() in ('true', '1', 'yes')
        txt_content = items.get('txt_content', None) if use_attr_text else None
        depth = items.get('depth', None)
        if depth is not None:
            depth = depth.to(device)
        results['cls_names'].extend(cls_name)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask.squeeze(1).numpy())  # px
        results['gt_sp'].extend(items['anomaly'])

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, args.features_list)
            use_depth = getattr(args, 'use_depth', False)
            use_regional = getattr(args, 'use_regional', True)
            if use_regional and use_depth and depth is not None:
                _, patch_tokens_depth = model.encode_depth(depth, args.features_list)
            else:
                patch_tokens_depth = None

            # 生成整体文本特征（不带属性）用于 anomaly_map_raw
            B, C = image_features.shape
            global_feat = image_features
            global_feat_new = New_Lan_Embed.prompt_linear1(global_feat.reshape(B, 1, C))
            
            prompt_query = New_Lan_Embed.prompt_query + torch.zeros(
                (B, New_Lan_Embed.prompt_query.shape[-2], New_Lan_Embed.prompt_query.shape[-1]), 
                dtype=New_Lan_Embed.prompt_query.dtype, 
                device=New_Lan_Embed.prompt_query.device
            )
            if args.use_global:
                class_token_global = prompt_query + global_feat_new
            else:
                class_token_global = prompt_query
            
            text_embeddings_global = prompt_ceshi.forward_ensemble(model, class_token_global, device, txt_content=None)
            text_embeddings_global = text_embeddings_global.permute(0,2,1)

            # 生成区域文本特征（带属性）用于 anomaly_map_new
            class_token_regional = New_Lan_Embed.before_extract_feat(patch_tokens,image_features.clone(), use_global = args.use_global)
            text_embeddings_regional = prompt_ceshi.forward_ensemble(model, class_token_regional, device, txt_content=txt_content)
            text_embeddings_regional = text_embeddings_regional.permute(0,2,1)
            
            anomaly_maps_new = []
            for layer in range(len(patch_tokens)):
                dense_feature = patch_tokens[layer][:,1:,:].clone()
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                if not use_regional:
                    anomaly_map_new = (New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings_regional)
                else:
                    if patch_tokens_depth is not None:
                        dense_depth = patch_tokens_depth[layer][:,1:,:].clone()
                        dense_depth = dense_depth / dense_depth.norm(dim=-1, keepdim=True)
                        F_t_a, _ = Zero_try(text_embeddings_regional.permute(0,2,1), dense_feature, F_s_depth=dense_depth)
                    else:
                        F_t_a, _ = Zero_try(text_embeddings_regional.permute(0,2,1), dense_feature, F_s_depth=None)
                    anomaly_map_new = (Zero_try.prompt_temp_l1.exp() * dense_feature @ F_t_a.permute(0,2,1))
                B, L, C = anomaly_map_new.shape 
                H = int(np.sqrt(L))
                anomaly_map_new = F.interpolate(anomaly_map_new.permute(0, 2, 1).view(B,2,H,H),
                                            size = args.image_size, mode = 'bilinear', align_corners=True)

                anomaly_map_new = torch.softmax(anomaly_map_new, dim =1)[:, 1, :, :]
                anomaly_maps_new.append(anomaly_map_new.cpu().numpy())

            patch_tokens_linear = linearlayer(patch_tokens)
            anomaly_maps_raw = []
            for layer in range(len(patch_tokens_linear)):
                dense_feature = patch_tokens_linear[layer].clone()
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                anomaly_map_raw = (New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings_global)
                B, L, C = anomaly_map_raw.shape 
                H = int(np.sqrt(L))
                anomaly_map_raw = F.interpolate(anomaly_map_raw.permute(0, 2, 1).view(B,2,H,H),
                                            size = args.image_size, mode = 'bilinear', align_corners=True)

                anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)[:, 1, :, :]
                anomaly_maps_raw.append(anomaly_map_raw.cpu().numpy())

            anomaly_map_raw = np.mean(anomaly_maps_raw, axis=0)
            anomaly_map_new = np.mean(anomaly_maps_new, axis=0)

            
            results['anomaly_map_raw'].append(anomaly_map_raw)
            results['anomaly_map_new'].append(anomaly_map_new)

    # metrics
    ap_px_ls_1 = []
    ap_px_ls_2 = []
    names_list = np.array(results['cls_names'])
    anomaly_maps_raw = np.concatenate(results['anomaly_map_raw'], axis=0)
    anomaly_maps_new = np.concatenate(results['anomaly_map_new'], axis=0)
    gts_pixel = np.concatenate(results['imgs_masks'], axis=0)
    for obj in obj_list:

        object_index = np.where(names_list == obj)[0]

        pr_px_1 = anomaly_maps_raw[object_index,:,:].copy()
        pr_px_2 = anomaly_maps_new[object_index,:,:].copy()

        gt_px = gts_pixel[object_index,:,:].copy()

        ap_px_1 = average_precision_score(gt_px.ravel(), pr_px_1.ravel())
        ap_px_2 = average_precision_score(gt_px.ravel(), pr_px_2.ravel())

        ap_px_ls_1.append(ap_px_1)
        ap_px_ls_2.append(ap_px_2)

    ap_mean_1 = np.mean(ap_px_ls_1)
    ap_mean_2 = np.mean(ap_px_ls_2)
    model.train()
    linearlayer.train()
    New_Lan_Embed.train()
    Zero_try.train()
    del results, gt_px, pr_px_1, pr_px_2
    return ap_mean_1, ap_mean_2
