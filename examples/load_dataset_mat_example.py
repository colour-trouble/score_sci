import torch
import argparse
import cv2

from utils.sci_dataloader import SCITrainingDatasetSubset, SCITestDataset
from utils.is_matrix_full_rank import *

### yaping: your path to the dataset
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--trainpath", default="/home/yaping/projects/data/SCI/matlab/")
# default="/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/")
parser.add_argument("--testpath", default="/home/yaping/projects/data/SCI/test_gray/")

args = parser.parse_args()

# yaping: files under the 'matlab/' folder
mask_location = args.trainpath + "mask.mat"
gt_location = args.trainpath + "gt/"
meas_location = args.trainpath + "measurement/"

batch_size = int(args.batch_size)
test_location = args.testpath

# yaping: dataloader
dataset = SCITrainingDatasetSubset(gt_location, meas_location, mask_location)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

test_dataset = SCITestDataset(test_location)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
)


# yaping: example enumerating training set
for ii, sample_batch in enumerate(dataloader):
    nimg = sample_batch["gt"].size(0)

    gt_batch = sample_batch["gt"].cuda()
    y = sample_batch["meas"].cuda()
    Phi = sample_batch["mask"].cuda()

    # Phi_sum = torch.sum(Phi, axis=3)
    # Phi_sum[Phi_sum == 0] = 1

    print("a random sample in the training set")
    print(gt_batch.shape, y.shape, Phi.shape)
    print("Is the mask full rank?")
    for i in range(Phi.shape[3]):
        print(is_full_rank(Phi[0, :, :, i]))

    break

# yaping: example enumerating testing set
print("Testing set")
for ii, sample_batch in enumerate(test_dataloader):

    gt_batch = sample_batch["gt"].cuda()
    y_batch = sample_batch["meas"].cuda()
    Phi = sample_batch["mask"].cuda()

    Phi_sum = torch.sum(Phi, axis=3)
    # Phi_sum[Phi_sum == 0] = 1
    viz_Phi = Phi_sum[0] / Phi.shape[3]

    print(sample_batch["file"])
    print(gt_batch.shape, y_batch.shape, Phi.shape)

# yaping: an example that visualizing the data
cv2.imshow(
    "ground truth", gt_batch[0, :, :, 0].cpu().numpy()
)  # the first(1st) frame of the ground truth video
cv2.imshow(
    "measurement", y_batch[0, :, :, 0].cpu().numpy()
)  # the first measurement (which compressing the 1st-8th frames of the video)
cv2.imshow("mask", viz_Phi.cpu().numpy())  # the mask

cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
