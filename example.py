import torch
from rt2.model import RT2

# img: (batch_size, 3, 256, 256)
# caption: (batch_size, 1024)
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

# model: RT2
model = RT2()

# Run model on img and caption
# Returns (logits, loss) tuple
logits, loss = model(img, caption)
print(f"Logits shape: {logits.shape}")  # (1, 1023, 20000)
print(f"Loss: {loss}")
