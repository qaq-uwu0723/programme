import sys
with open("D:/programme/tests/_torch_test.txt", "w") as f:
    f.write("starting...\n")
    f.flush()
    try:
        import torch
        f.write(f"torch OK, CUDA={torch.cuda.is_available()}\n")
    except Exception as e:
        f.write(f"ERROR: {e}\n")
    f.write("done\n")
