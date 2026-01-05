import torch

# 간단한 텐서 연산으로 GPU 확인
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print("GPU 연산 성공! 결과 shape:", z.shape)
print("GPU 메모리 사용량:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
print("뭉탱이")