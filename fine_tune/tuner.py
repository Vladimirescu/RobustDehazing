import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from fine_tune import DiffFitModel, LastLayerTune, TargetAdapt
from attacks import *


class FineTuningLightningModule(pl.LightningModule):
	def __init__(self, model, 
				  lr=1e-4, 
				  fine_tune_type="difffit",
				attack=None,
				attack_kwargs={},
				train_type="AT",
				train_type_kwargs={}):
		"""
		:param model: nn.Module, loaded with pre-trained weights
		:param fine_tune_type: string, on of the following:
			- last: only learns the final layer
			- first-last: first and last layers are learned
			- difffit: uses a similar fine-tuning to DiffFit used in diffusion
			- ...
		:param attack: class of attack to be used
		:param train_type: how to use adversarial samples during training
		:param train_type_kwargs: train_type-specific parameters
		"""
		super().__init__()
		self.model = model
		self.learning_rate = lr
		self.loss_fn = nn.L1Loss()  # Example loss function

		self.fine_tune_type = fine_tune_type
		self._prepare_model()
  
		if attack is None:
			self.attack = None
		else:
			self.attack = eval(attack)(self.model, **attack_kwargs)
		self.train_type = train_type

		self.tr_kwgs = train_type_kwargs

	def _prepare_model(self):
		if self.fine_tune_type == "difffit":
			self.model = DiffFitModel(self.model)
		elif self.fine_tune_type == "last":
			self.model = LastLayerTune(self.model)
		elif self.fine_tune_type == "adapt":
			self.model = TargetAdapt(self.model)
		else:
			raise ValueError(f"Unknown fine-tuning type {self.fine_tune_type}")

		n_layers = 0
		n_params = 0
		n_params_total = 0
		# Check n.o. trainables
		for p in self.model.parameters():
			n_params_total += p.numel()
			if p.requires_grad:
				n_layers += 1
				n_params += p.numel()
				
		print(f"Fine-tuning {n_layers} tensors, totaling {n_params} parameters ({n_params / n_params_total * 100} %).")

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch['source'], batch['target']
		  
		if self.train_type == "AT":
			x_ = self.attack(x, y)			
			y_pred = self(x_)
			loss = self.loss_fn(y_pred, y)
	
			self.log('train_loss', loss)
	
		elif self.train_type == "TRADES":
			x_ = self.attack(x, y)
	
			y_ = self(torch.cat([x, x_], dim=0))
			y_pred, y_pred_adv = torch.chunk(y_, chunks=2, dim=0)
      
			loss = self.loss_fn(y_pred, y)
			loss_adv = self.loss_fn(y_pred, y_pred_adv)
   
			loss = loss + self.tr_kwgs.lambd * loss_adv
   
			self.log("train_loss", loss)
			self.log("train_loss_adv", loss_adv)
  
		else:
			raise ValueError(f"Train type {self.train_type} not implemented.")
		  ###
		
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch['source'], batch['target']
								
		output = self(x).clamp_(-1, 1)	
   
		mse_loss = F.mse_loss(
			output * 0.5 + 0.5, y * 0.5 + 0.5, reduction='none'
		).mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
  
		self.log("mse_val", mse_loss.mean().item())
		self.log("psnr_val", psnr.item())

	def configure_optimizers(self):
		params = filter(lambda p: p.requires_grad, self.model.parameters())
		return torch.optim.Adam(params, lr=self.learning_rate)