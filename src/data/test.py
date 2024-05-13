import torch
import torchvision
import corda
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = corda.CORDA("/home/carloalberto/scratch/data-registry/medical/corda/releases")
    print("Loaded", len(dataset), "images")

    for item in dataset:
        print(*item[1:])

        print(item[0].min(), item[0].max(), item[0].shape)
        plt.figure()
        plt.imshow(item[0], cmap='bone')
        plt.show()