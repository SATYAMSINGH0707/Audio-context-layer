import torch


def main():
    print("=" * 60)
    print("Audio Context Layer - GPU Test")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("CUDA is NOT available.")
        return

    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda")

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    z = x @ y

    torch.cuda.synchronize()

    print("Matrix multiplication successful.")
    print("Result shape:", z.shape)
    print("Result device:", z.device)

    print("=" * 60)
    print("GPU TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
