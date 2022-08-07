import torch

from ApproximateFrequencySimilarityAttack.attack.DWT import DWT_audio, IDWT_audio


class LpDistance(object):
    def __init__(self, references, perturbed):
        self.noise = (perturbed - references).flatten(start_dim=1)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.references = references.to(self.device)
        self.perturbed = perturbed.to(self.device)

        self.DWT = DWT_audio(wavename='db6')
        self.IDWT = IDWT_audio(wavename='bd6')

    def Lp2(self):
        norm = torch.norm(self.noise, p=2, dim=-1)
        return torch.sum(torch.pow(norm, 2))

    def Lpinf(self):
        norm = torch.norm(self.noise, p=float('inf'), dim=-1)  # float('inf')
        return torch.sum(norm)

    def LowFreNorm(self):
        img_ll = self.DWT(self.references)
        img_ll = self.IDWT(img_ll)

        adv_ll = self.DWT(self.perturbed)
        adv_ll = self.IDWT(adv_ll)

        noise = (adv_ll - img_ll)
        norm = torch.norm(noise, p=2, dim=-1)
        return torch.sum(torch.pow(norm, 2))
