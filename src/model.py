from torchvision import datasets

dataset = datasets.ImageFolder("/Users/browka/Documents/squirell/data/raw")

print(dataset.classes)
print(dataset.class_to_idx)
