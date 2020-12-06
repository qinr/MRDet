import torch
from torch.autograd import gradcheck
from GConv import GOF_Function, GConv

def gradchecking(use_cuda=False):
    print('-'*80)
    GOF = GOF_Function.apply
    device = torch.device("cuda" if use_cuda else "cpu")

    weight = torch.randn(8,8,4,3,3).to(device).double().requires_grad_()
    gfb = torch.randn(4,3,3).to(device).double()

    test = gradcheck(GOF, (weight, gfb), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True)
    print(test)


if __name__ == "__main__":
    # gradchecking()
    # if torch.cuda.is_available():
    #     gradchecking(use_cuda=True)
    img = torch.randn(1, 256, 10, 10)
    gconv = GConv(256, 256, 3, padding=1, expand=True)
    out = gconv(img)
    print(out)

