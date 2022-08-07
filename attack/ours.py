import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from ApproximateFrequencySimilarityAttack.attack.DWT import DWT_audio_torch, IDWT_audio_torch


class OURS(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 steps: int = 150,
                 learning_rate: float = 0.001,
                 targeted: bool = False,
                 alpha: float = 1,
                 beta: float = 0.1,
                 wave_name: str = 'db6',
                 kappa: int = 0) -> None:
        super(OURS, self).__init__()
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = learning_rate
        self.target = targeted
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.wave_name = wave_name
        self.kappa = kappa

        self.fea_extract = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.fea_extract = nn.DataParallel(self.fea_extract)
        self.model = nn.DataParallel(self.model)

        self.DWT = DWT_audio_torch(self.wave_name)
        self.IDWT = IDWT_audio_torch(self.wave_name)

        self.targets = targeted

    def cal_sim(self, adv_fea, inputs_fea):
        adv_fea = F.normalize(adv_fea, dim=1)
        inputs_fea = F.normalize(inputs_fea, dim=1)

        r, c = inputs_fea.shape
        sim_matrix = torch.matmul(adv_fea, inputs_fea.T)
        mask = torch.eye(len(inputs_fea), dtype=torch.bool, device=self.device)
        pos_sim = sim_matrix[mask].view(r, -1)
        return pos_sim, sim_matrix

    def choose_target_similarity(self, pos_sim, sim_matrix):
        neg_sim = sim_matrix[:, -1].view(len(sim_matrix), -1)
        pos_neg_sim = torch.cat([pos_sim, neg_sim], dim=1)
        return pos_neg_sim

    def forward(self, inputs, labels):

        with torch.no_grad():
            inputs_fea = self.fea_extract(inputs)

        coeff = self.DWT(inputs)
        inputs_ll = self.IDWT(*coeff)

        w = torch.arctanh(inputs)
        w = Variable(w, requires_grad=True)
        w = w.to(self.device)
        optimizer = optim.Adam([w], lr=self.lr)

        best_adv_images = inputs.clone().detach()
        best_aux = 1e10 * torch.ones(len(inputs)).to(self.device)
        prev_cost = 1e10
        dim = len(inputs.shape)

        SmoothL1Loss = nn.SmoothL1Loss(reduction='none')

        for step in range(self.steps):
            optimizer.zero_grad()
            self.fea_extract.zero_grad()

            adv = torch.tanh(w)
            adv_fea = self.fea_extract(adv)

            coeff_adv = self.DWT(adv)
            adv_ll = self.IDWT(*coeff_adv)

            current_aux = SmoothL1Loss(adv_ll, inputs_ll).sum(dim=1)
            aux_loss = current_aux.sum()

            outputs = self.model(adv)

            pos_sim, sim_matrix = self.cal_sim(adv_fea, inputs_fea)
            pos_neg_sim = self.choose_target_similarity(pos_sim, sim_matrix)

            sim_neg = pos_neg_sim[:, -1]

            adv_loss = torch.sum(torch.clamp(1 - sim_neg + self.kappa, min=0))
            total_cost = self.beta * aux_loss + self.alpha * adv_loss
            # adv_cost = self.alpha * torch.clamp((sim_pos - sim_neg), min=0).sum()
            # total_cost = adv_cost

            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            common_id = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = (1 - common_id) * (best_aux > current_aux.detach())
            best_aux = mask * current_aux.detach() + (1 - mask) * best_aux

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if total_cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = total_cost.item()

        return best_adv_images
