import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.FC        = nn.Linear(256, 2)
		
	def forward(self, x, labels, track_change_start_frameind):
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			loss_all = self.criterion(x, labels)
			nloss = loss_all.mean()
			if loss_all.shape[0]>track_change_start_frameind:
				nloss_track_change = loss_all[track_change_start_frameind:].mean().detach().cpu().numpy()
			else:
				nloss_track_change = 0

			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, nloss_track_change, predScore, predLabel, correctNum

class lossA(nn.Module):
	def __init__(self):
		super(lossA, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)	
		nloss = self.criterion(x, labels)
		return nloss

class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()

		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)
		nloss = self.criterion(x, labels)
		return nloss

