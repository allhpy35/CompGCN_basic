from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
from sklearn.utils import class_weight


class BaseModel(torch.nn.Module):
	def __init__(self, params, class_weight=None):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh

		if params.task =='cls':
			# multi-class relation classification
			if hasattr(params, 'class_weight') and params.class_weight is not None:
				self.loss_fn = torch.nn.CrossEntropyLoss(
					weight=params.class_weight
				)
			else:
				self.loss_fn = torch.nn.CrossEntropyLoss()
		else:
			# link prediction (multi-label)
			self.loss_fn = torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.loss_fn(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base_lp(self, sub, rel, drop1, drop2):

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x

	def forward_base_cls(self, sub, obj, drop1, drop2):
		"""
        ✅ CLS 용: (sub, obj) -> sub_emb, obj_emb, all_ent
        - 핵심: message passing은 edge_type 기반으로 그대로 수행됨 (relation-aware)
        - 입력 rel이 없으므로 rel_emb index_select는 하지 않음
        """
		r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)

		x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x = drop1(x)
		x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x, r)
		x = drop2(x) if self.p.gcn_layer == 2 else x

		sub_emb = torch.index_select(x, 0, sub)
		obj_emb = torch.index_select(x, 0, obj)
		return sub_emb, obj_emb, x


class CompGCN_TransE(CompGCNBase):
	"""
		TransE의 점수는 원래: LP: 모든 엔티티 후보에 대해 score(h, r, t) = gamma - ||h + r - t||

		CLS는 “관계 후보 r_k” 들(5개)에 대해: logit_k = gamma - ||h + r_k - t||

	"""

	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

		if self.p.task == 'cls':
			# 5개 관계 클래스 임베딩
			self.cls_rel = torch.nn.Parameter(
				torch.randn(self.p.num_classes, self.p.embed_dim)
			)

	def forward(self, sub, rel):

		if self.p.task == 'cls':
			h, t , _ = self.forward_base_cls(sub, self.edge_type[sub], self.drop, self.drop)

			# logits: [B, C]
			# h: [B,d], r: [C,d], t:[B,d]
			# ||h + r - t|| -> [B,C]
			diff = (h.unsqueeze(1) + self.cls_rel.unsqueeze(0) - t.unsqueeze(1))
			logits = self.p.gamma - torch.norm(diff, p=1, dim=2)  # [B, C]
			return logits

		else:
			## Link Prediction(LP)
			sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
			obj_emb				= sub_emb + rel_emb
			x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
			score	= torch.sigmoid(x)
			return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

		if self.p.task == 'cls':
			# 분류용 relation embedding (5개 관계)
			self.cls_rel = torch.nn.Parameter(
				torch.randn(self.p.num_classes, self.p.embed_dim)
			)

	def forward(self, sub, re):
		"""
		LP  : forward(sub, rel)
        CLS : forward(sub, obj=obj)

		:param sub:
		:param re:
		:return:
		"""

		if self.p.task == 'cls':
			obj = re

			# CLS는 rel 필요 없음 → dummy rel
			dummy_rel = torch.zeros_like(sub)

			h, t, _ = self.forward_base_cls(sub, obj, self.drop, self.drop)

			# DistMult classifier
			# h: [B,d], cls_rel: [C,d], t: [B,d]
			logits = torch.sum(
				h.unsqueeze(1) *
				self.cls_rel.unsqueeze(0) *
				t.unsqueeze(1),
				dim=2
			)
			return logits

		else:
			# LP
			sub_emb, rel_emb, all_ent = self.forward_base(
				sub, re, self.drop, self.drop
			)

			obj_emb = sub_emb * rel_emb
			x = torch.mm(obj_emb, all_ent.transpose(1, 0))
			x += self.bias.expand_as(x)

			return torch.sigmoid(x)




class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz	= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

		if self.p.task == 'cls':
			self.cls_rel = torch.nn.Parameter(
				torch.randn(self.p.num_classes, self.p.embed_dim)
			)


	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		if self.p.task=='cls':
			obj = re
			dummy_rel = torch.zeros_like(sub)

			sub_emb, _, all_ent = self.forward_base(sub, dummy_rel, self.hidden_drop, self.feature_drop)
			obj_emb = all_ent[obj]  # [B,d]

			B, C, d = sub_emb.size(0), self.p.num_classes, sub_emb.size(1)

			# (B,C,d)로 확장해서 한번에 ConvE 수행
			sub_rep = sub_emb.unsqueeze(1).expand(B, C, d).reshape(B * C, d)
			rel_rep = self.cls_rel.unsqueeze(0).expand(B, C, d).reshape(B * C, d)

			x = self._conve_encode(sub_rep, rel_rep)  # [B*C, d]
			x = x.view(B, C, d)  # [B, C, d]

			# 각 클래스 출력 벡터 x_k 와 obj_emb dot => logits [B,C]
			logits = torch.sum(x * obj_emb.unsqueeze(1), dim=2)
			return logits

		else:
			sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
			stk_inp				= self.concat(sub_emb, rel_emb)
			x				= self.bn0(stk_inp)
			x				= self.m_conv1(x)
			x				= self.bn1(x)
			x				= F.relu(x)
			x				= self.feature_drop(x)
			x				= x.view(-1, self.flat_sz)
			x				= self.fc(x)
			x				= self.hidden_drop2(x)
			x				= self.bn2(x)
			x				= F.relu(x)

			x = torch.mm(x, all_ent.transpose(1,0))
			x += self.bias.expand_as(x)

			score = torch.sigmoid(x)
			return score
