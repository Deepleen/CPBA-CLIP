import torch.utils.data as data
import torch.utils.data.distributed
import json
from PIL import Image
import numpy as np
import torch
import os
import copy
import albumentations as A
import torchvision.transforms as transforms 


# CLIP's image normalization is applied to both RGB and depth inputs.  Depth
# images are loaded as 8-bit grayscale, so their fixed physical range is first
# mapped to [0, 1] before this normalization.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_NORMALIZE = transforms.Normalize(CLIP_MEAN, CLIP_STD)


def _build_augmentations():
	spatial_transform = A.Compose([
		A.Rotate(limit=30, p=0.5),
		A.RandomRotate90(p=0.5),
	], additional_targets={'depth': 'mask'}, is_check_shapes=False)
	rgb_transform = A.Compose([
		A.RandomBrightnessContrast(p=0.5),
		A.GaussNoise(p=0.5),
		A.OneOf([
			A.Blur(blur_limit=3, p=0.5),
			A.ColorJitter(p=0.5),
			A.GaussianBlur(p=0.5),
		], p=0.5),
	], is_check_shapes=False)
	return spatial_transform, rgb_transform


def _load_depth_image(root, img_path):
	base, ext = os.path.splitext(img_path)
	candidates = [base + '_depth' + ext]
	candidates.extend(base + '_depth' + candidate_ext for candidate_ext in ['.png', '.jpg', '.bmp', '.jpeg'])
	for relative_path in candidates:
		depth_path = os.path.join(root, relative_path)
		if os.path.exists(depth_path):
			try:
				return Image.open(depth_path).convert('L')
			except Exception:
				return None
	return None


def _augment_rgb_mask_depth(img, img_mask, depth, spatial_transform, rgb_transform):
	inputs = {'image': np.array(img), 'mask': np.array(img_mask)}
	if depth is not None:
		inputs['depth'] = np.array(depth)

	spatial_result = spatial_transform(**inputs)
	rgb_result = rgb_transform(image=spatial_result['image'])

	img = Image.fromarray(rgb_result['image'].astype(np.uint8))
	img_mask = Image.fromarray(spatial_result['mask'].astype(np.uint8), mode='L')
	if depth is not None:
		depth = Image.fromarray(spatial_result['depth'].astype(np.uint8), mode='L')
	return img, img_mask, depth


def _depth_to_tensor(depth, target_transform):
	"""Resize, scale, replicate, and CLIP-normalize a depth image.

	The depth files are read as 8-bit grayscale images.  ``ToTensor`` normally
	converts them to [0, 1]; the explicit range check also handles callers that
	pass an unscaled tensor (for example, values in [0, 255]).  The same scalar
	range is used for every image, rather than per-image min/max normalization,
	so depth magnitudes remain comparable across samples.
	"""
	if depth is None:
		return None
	depth = target_transform(depth) if target_transform is not None else transforms.ToTensor()(depth)
	if not isinstance(depth, torch.Tensor):
		depth = transforms.ToTensor()(depth)
	if depth.dim() == 2:
		depth = depth.unsqueeze(0)
	if depth.dim() != 3:
		raise ValueError(
			f"Expected depth tensor with shape [C, H, W], got {tuple(depth.shape)}"
		)

	depth = depth.float()
	if not torch.isfinite(depth).all():
		raise ValueError("Depth tensor contains NaN or infinite values")
	if depth.numel() > 0 and float(depth.detach().amax()) > 1.0:
		depth = depth / 255.0
	depth = depth.clamp(0.0, 1.0)

	if depth.shape[0] == 1:
		depth = depth.repeat(3, 1, 1)
	elif depth.shape[0] != 3:
		raise ValueError(
			f"Expected a grayscale or 3-channel depth image, got {depth.shape[0]} channels"
		)

	# Apply exactly the same mean/std as the RGB CLIP preprocessing pipeline.
	return _CLIP_NORMALIZE(depth)


def _build_sample(root, img_path, img, img_mask, cls_name, anomaly, txt_content, depth, use_depth):
	"""Build a collatable sample while keeping optional modalities explicit."""
	sample = {
		'img': img,
		'img_mask': img_mask,
		'cls_name': cls_name,
		'anomaly': anomaly,
		'img_path': os.path.join(root, img_path),
		# Missing attribute files are represented by an empty string, not None.
		'txt_content': txt_content or '',
	}
	if use_depth:
		if depth is None:
			raise RuntimeError(
				f"Depth image is missing or unreadable for RGB image: "
				f"{os.path.join(root, img_path)}"
			)
		sample['depth'] = depth
	return sample

DEFAULT_VISA_VALIDATION_PRODUCTS = [
	"chewinggum", "cashew", "pipe_fryum", "capsules", "candle"
]
DEFAULT_MVTEC_VALIDATION_PRODUCTS = [
	"bottle", "hazelnut", "wood", "zipper", "leather"
]


def Split_Product(train_dataset, val_dataset=None, val_product_list=None):
	"""Build the single cross-dataset split used by training.

	All categories from the selected training dataset are used. ``pre=None``
	deliberately means the dataset loader keeps every category in the matching
	meta file. If no validation dataset is supplied, the other supported dataset
	is selected automatically; ``val_product_list`` can override its categories.
	"""
	available_datasets = {"visa", "mvtec"}
	if train_dataset not in available_datasets:
		raise ValueError(
			f"Unsupported training dataset: {train_dataset!r}. "
			"Expected 'visa' or 'mvtec'."
		)

	if val_dataset is None:
		val_dataset = "mvtec" if train_dataset == "visa" else "visa"
	if val_dataset not in available_datasets:
		raise ValueError(
			f"Unsupported validation dataset: {val_dataset!r}. "
			"Expected 'visa' or 'mvtec'."
		)
	if train_dataset == val_dataset:
		raise ValueError(
			"Training and validation datasets must be different for the "
			"cross-dataset protocol."
		)

	default_validation_products = (
		DEFAULT_VISA_VALIDATION_PRODUCTS
		if val_dataset == "visa" else DEFAULT_MVTEC_VALIDATION_PRODUCTS
	)
	validation_products = (
		list(default_validation_products)
		if val_product_list is None else list(val_product_list)
	)
	if not validation_products:
		raise ValueError(
			f"val_product_list must contain at least one {val_dataset} category."
		)

	return [dict(
		train_dataset=train_dataset,
		pre=None,
		val_dataset=val_dataset,
		post=validation_products,
	)]

		


class Makedataset():
	def __init__(self, train_data_path, preprocess_test, mode, train_mode, image_size = 518, use_depth = False):
		self.train_data_path= train_data_path
		self.preprocess_test = preprocess_test
		self.mode = mode
		self.train_mode = train_mode
		self.target_transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.CenterCrop(image_size),transforms.ToTensor()])
		self.shuf = True if self.mode == "train" else False
		self.use_depth = use_depth

	def mask_dataset(self, name, product_list, batchsize, shuf = True, sampler=None, drop_last=False):
		if name not in {"mvtec", "visa"}:
			raise ValueError(f"Unsupported dataset: {name!r}. Expected 'mvtec' or 'visa'.")
		if product_list is not None:
			meta_path = os.path.join(self.train_data_path, f"meta_{name}.json")
			with open(meta_path, 'r') as meta_file:
				available_products = set(json.load(meta_file)["test"].keys())
			unknown_products = sorted(set(product_list) - available_products)
			if unknown_products:
				raise ValueError(
					f"Unknown {name} categories: {unknown_products}. "
					f"Available categories: {sorted(available_products)}"
				)
		if name == "mvtec" and self.train_mode == "zero":
			dataset = MVTecDataset(root=self.train_data_path, transform=self.preprocess_test, target_transform=self.target_transform,
								  mode =self.mode, train_mode=self.train_mode, product_list= product_list, dataset="mvtec", use_depth=self.use_depth)
		elif name == "visa" and self.train_mode == "zero":
			dataset = VisaDataset(root=self.train_data_path, transform=self.preprocess_test, target_transform=self.target_transform,
											   mode =self.mode, train_mode=self.train_mode, product_list= product_list, dataset= "visa", use_depth=self.use_depth)
		obj_list = dataset.get_cls_names()
		
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batchsize,
			shuffle=shuf if sampler is None else False,
			sampler=sampler,
			drop_last=drop_last
		)
		
		return dataloader, obj_list
		

			



def merge(di1, di2):
	he_bing = copy.deepcopy(di1)
	for di2_key in di2.keys():
		if len(di2[di2_key]) > 1:
			if di2_key not in di1.keys():
				he_bing[di2_key] = di2[di2_key]
			else:
				he_bing[di2_key].extend(di2[di2_key])
	return he_bing


class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='train', train_mode = 'zero', aug = True, product_list = None, dataset = "visa", use_depth = False):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug = aug
		self.mode = mode
		self.train_mode = train_mode
		self.use_depth = use_depth

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_{dataset}.json', 'r'))
		meta_info = meta_info["test"]
		
		if train_mode == "zero" and product_list is not None:
			keys = meta_info.keys()
			keys = list(keys)
			for product in product_list:
				assert product in keys
			for key in keys:
				if key not in product_list:
					del meta_info[key]

		self.spatial_transform, self.rgb_transform = _build_augmentations()

		self.cls_names = list(meta_info.keys())
		
		for index, cls_name in enumerate(self.cls_names):
			self.data_all.extend(meta_info[cls_name])
		
		#self.data_all = self.data_all[:200]
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		
		# 读取 txt 文件内容
		txt_content = None
		if img_path:
			# 构建 txt 文件路径：将图片扩展名替换为 .txt
			txt_path = os.path.splitext(img_path)[0] + '.txt'
			full_txt_path = os.path.join(self.root, txt_path)
			if os.path.exists(full_txt_path):
				try:
					with open(full_txt_path, 'r', encoding='utf-8') as f:
						txt_content = f.read().strip()
				except Exception as e:
					# 如果读取失败，txt_content 保持为 None
					pass
		
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		depth_image = _load_depth_image(self.root, img_path) if self.use_depth and img_path else None
		if self.mode == "train" and self.aug == True:
			img, img_mask, depth_image = _augment_rgb_mask_depth(
				img, img_mask, depth_image, self.spatial_transform, self.rgb_transform
			)
		
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		depth = _depth_to_tensor(depth_image, self.target_transform)

		return _build_sample(
			self.root, img_path, img, img_mask, cls_name, anomaly,
			txt_content, depth, self.use_depth
		)


class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='train', train_mode = 'zero', aug = True, dataset = "mvtec", product_list = None, use_depth = False):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.mode = mode
		self.train_mode =  train_mode
		self.aug = aug
		self.use_depth = use_depth
		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_{dataset}.json', 'r'))
		meta_info = meta_info["test"]
		
		if train_mode == "zero" and product_list is not None:
			keys = meta_info.keys()
			keys = list(keys)
			for product in product_list:
				assert product in keys
			for key in keys:
				if key not in product_list:
					del meta_info[key]
		
		self.spatial_transform, self.rgb_transform = _build_augmentations()


		self.cls_names = list(meta_info.keys())
				
		for index, cls_name in enumerate(self.cls_names):
			self.data_all.extend(meta_info[cls_name])
		
		#self.data_all = self.data_all[:200]
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		
		# 读取 txt 文件内容
		txt_content = None
		if img_path:
			# 构建 txt 文件路径：将图片扩展名替换为 .txt
			txt_path = os.path.splitext(img_path)[0] + '.txt'
			full_txt_path = os.path.join(self.root, txt_path)
			if os.path.exists(full_txt_path):
				try:
					with open(full_txt_path, 'r', encoding='utf-8') as f:
						txt_content = f.read().strip()
				except Exception as e:
					# 如果读取失败，txt_content 保持为 None
					pass
		
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		depth_image = _load_depth_image(self.root, img_path) if self.use_depth and img_path else None
		if self.mode == "train" and self.aug == True:
			img, img_mask, depth_image = _augment_rgb_mask_depth(
				img, img_mask, depth_image, self.spatial_transform, self.rgb_transform
			)
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		depth = _depth_to_tensor(depth_image, self.target_transform)

		return _build_sample(
			self.root, img_path, img, img_mask, cls_name, anomaly,
			txt_content, depth, self.use_depth
		)


class OtherDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='test', train_mode = 'zero', aug = True, product_list = None, dataset = None, use_depth = False):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug = aug
		self.mode = mode
		self.train_mode = train_mode
		self.use_depth = use_depth

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_{dataset}.json', 'r'))
		meta_info = meta_info["test"]
		
		if train_mode == "zero" and product_list is not None:
			keys = meta_info.keys()
			keys = list(keys)
			for product in product_list:
				assert product in keys
			for key in keys:
				if key not in product_list:
					del meta_info[key]

		self.spatial_transform, self.rgb_transform = _build_augmentations()

		self.cls_names = list(meta_info.keys())
		
		for index, cls_name in enumerate(self.cls_names):
			self.data_all.extend(meta_info[cls_name])
		
		#self.data_all = self.data_all[:200]
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		
		# 读取 txt 文件内容
		txt_content = None
		if img_path:
			# 构建 txt 文件路径：将图片扩展名替换为 .txt
			txt_path = os.path.splitext(img_path)[0] + '.txt'
			full_txt_path = os.path.join(self.root, txt_path)
			if os.path.exists(full_txt_path):
				try:
					with open(full_txt_path, 'r', encoding='utf-8') as f:
						txt_content = f.read().strip()
				except Exception as e:
					# 如果读取失败，txt_content 保持为 None
					pass
		
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		depth_image = _load_depth_image(self.root, img_path) if self.use_depth and img_path else None
		if self.mode == "train" and self.aug == True:
			img, img_mask, depth_image = _augment_rgb_mask_depth(
				img, img_mask, depth_image, self.spatial_transform, self.rgb_transform
			)
		
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		depth = _depth_to_tensor(depth_image, self.target_transform)

		return _build_sample(
			self.root, img_path, img, img_mask, cls_name, anomaly,
			txt_content, depth, self.use_depth
		)
