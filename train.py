import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import  json 
import argparse 
import numpy as np 
import random 
import os 
import torch 
from torch import nn 
from torch.nn import functional as F 
import torchvision.transforms as transforms 
import logging 
from models.model_CLIP import Load_CLIP, tokenize
from models.lora import inject_lora_to_depth_visual
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from utils.dataset import Makedataset, Split_Product
from utils.loss import FocalLoss, BinaryDiceLoss
from models.prompt_ensemble import Prompt_Ensemble
from tqdm import tqdm
from models.CPG import Context_Prompting 
from models.BAN import Zero_Parameter  
from utils.evaluate import evaluate_post
import copy

DATASET_LABELS = {"visa": "VisA", "mvtec": "MVTec"}


def _get_lora_variant(args):
    if getattr(args, 'use_rslora', False) and getattr(args, 'use_dora', False):
        raise ValueError("--use_rslora and --use_dora cannot be enabled at the same time.")
    if getattr(args, 'use_dora', False):
        return "DoRA"
    if getattr(args, 'use_rslora', False):
        return "RSLoRA"
    return "LoRA"

def _load_stages(model, params, exclude_key=None):  # Load the weights of learnable prompts (and lora when resume).
    for n, m in model.named_parameters():
        if exclude_key is None:
            continue
        if isinstance(exclude_key, str):
            match = exclude_key in n
        else:
            match = any(ek in n for ek in exclude_key)
        if match and n in params:
            assert m.data.size() == params[n].data.size()
            m.data = params[n].data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

class LinearLayer(nn.Module): # linear layers used for mapping patch-level features.
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        assert 'ViT' in model
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
    def forward(self, tokens):
        tokens_list = []
        for i in range(len(tokens)):
            tokens_list.append(self.fc[i](tokens[i][:, 1:, :]))
        return tokens_list

def _freeze_stages(model, exclude_key=None):  # Freeze all parameters except for the learnable prompts. (All the parameters that need to be trained in this code have "prompt" in their names.)
    """Freeze stages param and norm stats."""
    parameter_prompt_dict = {}
    for n, m in model.named_parameters():
        if exclude_key:
            if isinstance(exclude_key, str):
                if not exclude_key in n:  
                    m.requires_grad = False
                else:
                    parameter_prompt_dict[n] = m
            elif isinstance(exclude_key, list):
                count = 0
                for i in range(len(exclude_key)):
                    i_layer = str(exclude_key[i])
                    if i_layer in n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count>0:  
                    print('Finetune layer in backbone:', n)
                    # Keep matching prompt/LoRA parameters in the optimizer and checkpoint.
                    parameter_prompt_dict[n] = m
            else:
                assert AttributeError("Dont support the type of exclude_key!")
        else:
            m.requires_grad = False
    return parameter_prompt_dict


def train(args):

    image_size = args.image_size 
    epochs = args.epoch
    tokenizer = tokenize
    learning_rate = args.learning_rate 
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path,"result.txt")
    features_list  = args.features_list 
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    
    #---------------- start writing logs ----------------------#
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_hander  = logging.FileHandler(log_path, mode = 'a+')
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)
    console_hander = logging.StreamHandler()
    console_hander.setFormatter(formatter)
    logger.addHandler(console_hander)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args,arg)}')
    #---------------- end writing logs ----------------------#


    model , preprocess_train , preprocess_test = Load_CLIP(image_size, args.pretrained_path , device=device, deep_prompt_len = args.deep_prompt_len, total_d_layer_len = args.total_d_layer_len)

    # Inject LoRA into depth_visual branch if requested
    if args.use_lora and args.use_depth:
        lora_variant = _get_lora_variant(args)
        logger.info(
            f"Injecting {lora_variant} (r={args.lora_r}, alpha={args.lora_alpha}) into depth_visual blocks: {args.lora_blocks}"
        )
        model = inject_lora_to_depth_visual(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            target_blocks=args.lora_blocks,
            use_rslora=args.use_rslora,
            use_dora=args.use_dora,
        )

    model.to(device)
    exclude_keys = ["prompt", "lora_"] if (args.use_lora and args.use_depth) else "prompt"
    model_optim = _freeze_stages(model, exclude_keys)

    Make_dataset = Makedataset(train_data_path = args.train_data_path , preprocess_test = preprocess_test, mode = "train", 
                               train_mode = "zero", image_size = args.image_size, use_depth = getattr(args, 'use_depth', False))

    Make_dataset_val = Makedataset(train_data_path = args.val_data_path , preprocess_test = preprocess_test, mode = "test", 
                               train_mode = "zero", image_size = args.image_size, use_depth = getattr(args, 'use_depth', False))

    Product_groups = Split_Product(
        args.dataset, args.val_dataset, args.val_product_list
    )




    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(args.features_list), args.model).to(device)
    trainable_layer.train()

    New_Lan_Embed = Context_Prompting(model_configs, cla_len=args.prompt_len).to(device)
    New_Lan_Embed.train()
    prompt_pre = Prompt_Ensemble(args.prompt_len,tokenizer)  # Generate the initial text embeddings

    Zero_try = Zero_Parameter(dim_v = model_configs["vision_cfg"]['width'], dim_t =  model_configs['text_cfg']['width'], dim_out= model_configs["vision_cfg"]['width']).to(device)
    Zero_try.train()   # Further update text embeddings in the post-VCP module using detailed patch-level features


    parameter_prompt_list = []
    for n, m in New_Lan_Embed.named_parameters():
        if n != "prompt_temp": #prompt_temp is learnable temperature coefficient $\tau_1$
            parameter_prompt_list.append(m)
        else:
            print(n)

    group1 = []
    group2 = []
    for n, m in Zero_try.named_parameters():
        if n not in ["prompt_temp_l1"]: # "prompt_temp_l1 is learnable temperature coefficient $\tau_2$
            group1.append(m)
        else:
            group2.append(m)
            print(n)


    parameter_model_prompt_list = [value for key,value in model_optim.items()]

    lr_group1 = list(trainable_layer.parameters()) +  parameter_prompt_list +  parameter_model_prompt_list
    lr_group2 = [New_Lan_Embed.prompt_temp] 

    optimizer = torch.optim.Adam([{'params': group1 + lr_group1, 'lr': args.learning_rate}, {'params': group2 + lr_group2, 'lr':0.01}], lr = learning_rate, betas = (0.5 , 0.999))  
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()



    for product_split in Product_groups:
        logger.info(
            "Training on all %s categories; validating on %s categories: %s",
            DATASET_LABELS[product_split["train_dataset"]],
            DATASET_LABELS[product_split["val_dataset"]],
            product_split["post"],
        )
        pre_dataloader, pre_obj_list = Make_dataset.mask_dataset(
            name=product_split["train_dataset"], product_list=product_split["pre"],
            batchsize=args.batch_size, shuf=True
        )
        val_dataloader, val_obj_list = Make_dataset_val.mask_dataset(
            name=product_split["val_dataset"], product_list=product_split["post"],
            batchsize=1, shuf=False
        )

        if args.resume_checkpoint_path is not None:  # Resume training from a checkpoint
            checkpoint = torch.load(args.resume_checkpoint_path, map_location= device)
            trainable_layer.load_state_dict(checkpoint["trainable_linearlayer"])
            Zero_try.load_state_dict(checkpoint["Zero_try"])
            New_Lan_Embed.load_state_dict(checkpoint["New_Lan_Embed"])
            _load_stages(model, checkpoint, exclude_keys)


        ap_global_max = 0
        ap_px_max = 0
        
        for epoch in range(epochs):
            loss_list = []
            loss_raw_list = []
            loss_new_list = []
            idx = 0
            pre_dataloader = tqdm(pre_dataloader)
            for items in pre_dataloader:
                idx += 1
                image = items['img'].to(device)
                cls_name = items['cls_name']
                use_attr_text = getattr(args, 'use_attribute_text', True)
                if isinstance(use_attr_text, str):
                    use_attr_text = use_attr_text.lower() in ('true', '1', 'yes')
                txt_content = items.get('txt_content', None) if use_attr_text else None  # 属性级描述文本由 --use_attribute_text 控制
                depth = items.get('depth', None)
                if depth is not None:
                    depth = depth.to(device)
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        image_features, patch_tokens = model.encode_image(image, features_list)
                    if args.use_regional and args.use_depth and depth is not None:
                        _, patch_tokens_depth = model.encode_depth(depth, features_list)
                    else:
                        patch_tokens_depth = None
                    
                    class_token_global = New_Lan_Embed.before_extract_feat(patch_tokens,image_features, use_global = args.use_global)
                    text_embeddings_global = prompt_pre.forward_ensemble(model, class_token_global, device, txt_content=None)
                    text_embeddings_global = text_embeddings_global.permute(0,2,1)

                    class_token_regional = New_Lan_Embed.before_extract_feat(patch_tokens,image_features, use_global = False)
                    text_embeddings_regional = prompt_pre.forward_ensemble(model, class_token_regional, device, txt_content=txt_content) 

                    text_embeddings_regional = text_embeddings_regional.permute(0,2,1)

                    anomaly_maps_new = []
                    for layer in range(len(patch_tokens)):
                        dense_feature = patch_tokens[layer][:,1:,:].clone()
                        dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                        if not args.use_regional:
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
                                                    size = image_size, mode = 'bilinear', align_corners=True)

                        anomaly_map_new = torch.softmax(anomaly_map_new, dim =1)
                        anomaly_maps_new.append(anomaly_map_new)
                    
                    patch_tokens_linear = trainable_layer(patch_tokens)
                    anomaly_maps_raw = []
                    for layer in range(len(patch_tokens_linear)):
                        dense_feature = patch_tokens_linear[layer].clone()
                        dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                        anomaly_map_raw = (New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings_global)
                        B, L, C = anomaly_map_raw.shape 
                        H = int(np.sqrt(L))
                        anomaly_map_raw = F.interpolate(anomaly_map_raw.permute(0, 2, 1).view(B,2,H,H),
                                                    size = image_size, mode = 'bilinear', align_corners=True)

                        anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)
                        anomaly_maps_raw.append(anomaly_map_raw)


                gt = items['img_mask'].squeeze().to(device)
                gt[gt > 0.5], gt[gt< 0.5] = 1, 0
                loss_new = 0
                for num in range(len(anomaly_maps_new)):
                    loss_new += loss_focal(anomaly_maps_new[num], gt)
                    loss_new += loss_dice(anomaly_maps_new[num][:, 1, :, :], gt)

                loss_base = 0
                for num in range(len(anomaly_maps_raw)):
                    loss_base += loss_focal(anomaly_maps_raw[num], gt)
                    loss_base += loss_dice(anomaly_maps_raw[num][:, 1, :, :], gt)

                loss = loss_new + loss_base
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                loss_raw_list.append(loss_base.item())
                loss_new_list.append(loss_new.item())
                print(loss.item(), loss_base.item(),loss_new.item())
                del patch_tokens_linear, patch_tokens, dense_feature, anomaly_map_raw, anomaly_map_new, loss, loss_new, loss_base
                if patch_tokens_depth is not None:
                    del patch_tokens_depth
                torch.cuda.empty_cache()

            #-------------------------------Start Evaluation ------------------------------------#
            ap_px, ap_regional, ap_global = evaluate_post(val_dataloader, model, trainable_layer, New_Lan_Embed, Zero_try, prompt_pre, device, args, val_obj_list)
            if ap_px > ap_px_max:
                logger.info('epoch [{}/{}], ap_px_max update:{:.4f}, ap_regional:{:.4f}, ap_global:{:.4f}'.format(epoch + 1, epochs, ap_px, ap_regional, ap_global))
                ap_px_max = ap_px
            if ap_global > ap_global_max:
                ap_global_max = ap_global
                logger.info('epoch [{}/{}], ap_px_max:{:.4f}, ap_regional:{:.4f}, ap_global_max update:{:.4f}'.format(epoch + 1, epochs, ap_px, ap_regional, ap_global))

            if (epoch + 1) % args.print_freq == 0:
                logger.info('epoch [{}/{}], loss:{:.4f}  loss_base:{:.4f}   loss_new:{:.4f}  ap_px:{:.4f}  ap_regional:{:.4f}  ap_global:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list),np.mean(loss_raw_list),np.mean(loss_new_list), ap_px, ap_regional, ap_global))
            #---------------------------- End evaluation ----------------------------------------#
            # save model
            if (epoch + 1) % args.save_freq == 0:
                ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
                save_dict = {'trainable_linearlayer': trainable_layer.state_dict(), 'New_Lan_Embed': New_Lan_Embed.state_dict(), "Zero_try":Zero_try.state_dict()}
                save_dict.update(model_optim)
                torch.save(save_dict, ckp_path)


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser("CPBA-CLIP", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./dataset/mvisa/data", help="train dataset path")
    parser.add_argument("--val_data_path", type=str, default="./dataset/mvisa/data", help="val dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/CPBA_clip_train_mvtec', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./models/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="weight path of CLIP model")
    parser.add_argument("--resume_checkpoint_path", type=str, default= None, help="resume_checkpoint_path")
    # model
    parser.add_argument("--dataset", type=str, choices=["visa", "mvtec"], default="mvtec",
                        help="training dataset (default: visa)")
    parser.add_argument("--val_dataset", type=str, choices=["visa", "mvtec"], default="visa",
                        help="validation dataset; defaults to the other supported dataset")
    parser.add_argument("--val_product_list", type=str, nargs="+", default=None,
                        help="categories used for validation; defaults to a configured five-category split")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # training
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--device_id", type=int, default=0, help="GPU id: >=0")
    parser.add_argument("--seed", type=int, default= 333, help="random seed")

    # hyper-parameter
    parser.add_argument("--prompt_len", type=int, default=2, help="the length of the learnable category vectors r")
    parser.add_argument("--deep_prompt_len", type=int, default=1, help="the length of the learnable text embeddings n ")
    parser.add_argument("--use_global", default=True, action="store_false", help="Whether to use the CPG")
    parser.add_argument("--use_regional", default=True, action="store_false", help="Whether to use the BAN ")
    parser.add_argument("--total_d_layer_len", type=int, default= 11, help="number of layers for the text encoder with learnable text embeddings")

    # Attribute-level description, depth map, and LoRA settings
    parser.add_argument("--use_attribute_text", default=False, help="use attribute-level description text for prompts (like --use_depth for depth)")
    parser.add_argument("--use_depth",  default=False, help="use depth image as third modality (RGB + depth + text)")
    parser.add_argument("--use_lora", default=False, help="inject LoRA into depth_visual branch when use_depth")
    parser.add_argument("--use_rslora", default=False, help="use RSLoRA scaling on top of LoRA")
    parser.add_argument("--use_dora", default=False, help="use DoRA variant on top of LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora_blocks", type=int, nargs="+", default=[-1, -2, -3, -4], help="depth_visual block indices for LoRA (negative = from end)")

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    setup_seed(args.seed)
    train(args)
    
