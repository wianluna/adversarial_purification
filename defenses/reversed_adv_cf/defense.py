from defenses.reversed_adv_cf.adv_cf import attack

bounds = {
    "low": -6.302890777587893,
    "high": 137.54115295410156
}


class ReversedAdvCF:
    def __init__(self, model):
        self.model = model

    def __call__(self, image):
        return attack(image, model=self.model, metric_range=bounds['high'] - bounds['low'], device='cuda')
