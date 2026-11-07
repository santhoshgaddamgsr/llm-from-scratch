import torch 
print(torch.__version__, torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device_name:", torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
print(x.device)
print(x*2)