import torch

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.Linear(28 * 28, 10).to(device)
    print(model)
