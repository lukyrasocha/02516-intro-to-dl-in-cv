import pickle
import matplotlib as mpl
from load_data import Potholes
from torchvision import transforms
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from visualize import visualize_proposals
from tensordict import TensorDict
from metrics import IoU

################################################################
### move later to visualize.py

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'text.usetex': True})

# Set global font size for title, x-label, and y-label
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16

# Set global font size for x and y tick labels
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Set global font size for the legend
plt.rcParams['legend.fontsize'] = 11

# Set global font size for the figure title
plt.rcParams['figure.titlesize'] = 42

color_primary = '#990000'  # University red


################################################################


def generate_proposals_selective_search(image_tensor, max_proposals=5000, type='fast'):
    """
    Generates proposals using the Selective Search algorithm.

    Args:
        image_tensor (Tensor): The image tensor for which to generate proposals.
        max_proposals (int): The maximum number of proposals to return.
        type (str): The type of Selective Search to use. Options are 'fast' and 'quality'.

    Returns:
        list: A list of TensorDict objects, each containing bounding box coordinates.
    """
    # Convert tensor to a NumPy array (HxWxC format)
    image_np = np.array(to_pil_image(image_tensor))

    # Initialize selective search
    selection_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selection_search.setBaseImage(image_np)

    if type == 'fast':
        selection_search.switchToSelectiveSearchFast()
    elif type == 'quality':
        selection_search.switchToSelectiveSearchQuality()

    # Run selective search to get bounding boxes
    rects = selection_search.process()

    # Limit the number of proposals
    rects = rects[:max_proposals]

    # Convert rects to proposals in TensorDict format
    proposals = []
    for (x, y, w, h) in rects:
        proposal = {
            'xmin': torch.tensor(float(x)),
            'ymin': torch.tensor(float(y)),
            'xmax': torch.tensor(float(x + w)),
            'ymax': torch.tensor(float(y + h)),
            'labels': torch.tensor(-1, dtype=torch.int64)  # -1 for unlabeled
        }
        proposals.append(proposal)

    return proposals


def calculate_recall(proposals, ground_truth_boxes, iou_threshold):
    """
    Calculates recall for a given IoU threshold using all proposals
    and stores all matches for each ground truth box in a dictionary, including coordinates.

    Args:
        proposals (list): List of TensorDict objects containing proposal bounding box coordinates.
        ground_truth_boxes (list): List of TensorDict objects containing ground truth bounding box coordinates.
        iou_threshold (float): IoU threshold to consider a proposal as a match.

    Returns:
        float: Recall for the specified IoU threshold.
        dict: Dictionary where each key is a ground truth box index and each value is a list of matched proposals,
              with proposal index, IoU, and bounding box coordinates.
        list: List of all unique proposals that have at least one IoU match above the threshold.
    """
    matches = {}
    matched_gt_boxes = set()
    matching_proposals = set()  #Store unique proposal indices that match any gt box 
                                # sets also also much faster and make sense to use here
                                # similar to mathematical sets 

    # Loop through each proposal
    for prop_idx, proposal in enumerate(proposals):
        proposal_bbox = {
            'xmin': proposal['xmin'].item(),
            'ymin': proposal['ymin'].item(),
            'xmax': proposal['xmax'].item(),
            'ymax': proposal['ymax'].item()
        }

        # Track matches for each ground truth box
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            gt_bbox = {
                'xmin': gt_box['xmin'].item(),
                'ymin': gt_box['ymin'].item(),
                'xmax': gt_box['xmax'].item(),
                'ymax': gt_box['ymax'].item()
            }
            iou = IoU(proposal_bbox, gt_bbox)

            # Add to matches if IoU is above threshold
            if iou >= iou_threshold:
                if gt_idx not in matches:
                    matches[gt_idx] = []
                matches[gt_idx].append({
                    'proposal_idx': prop_idx,
                    'iou': iou,
                    'proposal_bbox': proposal_bbox
                })
                matched_gt_boxes.add(gt_idx)
                matching_proposals.add(prop_idx)  # Add to the set of matching proposals

    # Calculate recall
    recall = len(matched_gt_boxes) / len(ground_truth_boxes) if ground_truth_boxes else 0.0

    # Convert matching_proposals set to a list and retrieve corresponding proposals
    all_matching_proposals = [proposals[idx] for idx in sorted(matching_proposals)]

    return recall, matches, all_matching_proposals


def visualize_proposals_and_gt(image, ground_truth, proposals, iou_threshold):
    # Convert the image tensor to a format compatible with matplotlib
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # Change dimensions from CxHxW to HxWxC

    # Create the plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f'Image with Ground Truth and Proposal Bounding Boxes with IoU {iou_threshold}', color=color_primary)

    # Plot ground truth bounding boxes in green
    for gt_box in ground_truth:
        xmin, ymin, xmax, ymax = gt_box['xmin'], gt_box['ymin'], gt_box['xmax'], gt_box['ymax']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, 'Ground Truth', color='green', fontsize=12)

    # Plot proposal bounding boxes in red
    for proposal in proposals:
        xmin, ymin, xmax, ymax = proposal['xmin'], proposal['ymin'], proposal['xmax'], proposal['ymax']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, 'Proposal', color='red', fontsize=10)

    plt.axis('off')
    plt.savefig('proposals_ground_truth.png')
    plt.show()



def generate_proposals_for_entire_dataset(iou_threshold, num_images, pickle=False, quality='fast'):
    
    # Initialize lists to store all proposals and ground truth boxes
    all_image_proposals = []


    for i in range(num_images):
        # Load image and ground truth boxes
        #index = i + np.random.randint(0, 500)
        #index = 48
        index = 75
        image, target = potholes_dataset[index]

        # the index is 
        print(f"The index is {index}")
        
        ground_truth_count = len(target)
        
        # Generate proposals using selective search
        proposals = generate_proposals_selective_search(image, max_proposals=9000, type=quality)
        
        # Calculate recall and get all matching proposals
        recall, matches, all_matching_proposals = calculate_recall(proposals, target, iou_threshold)
        
        # Store the matching proposals in the desired format
        image_proposals = []
        for proposal in all_matching_proposals:
            proposal_dict = TensorDict({
                'xmin': proposal['xmin'],
                'ymin': proposal['ymin'],
                'xmax': proposal['xmax'],
                'ymax': proposal['ymax'],
                'labels': torch.tensor(1, dtype=torch.int64) 
            })
            image_proposals.append(proposal_dict)
        
        # Append to the list of all proposals
        all_image_proposals.append({
            'image': image,
            'proposals': image_proposals,
            'ground_truhs': target,  
        })
        
    # added a pickle, since it might be smart to do this only once instead of every time
    # we run the model 
    if pickle:
        with open(f'proposals_iou_{iou_threshold}.pkl', 'wb') as f:
            pickle.dump(all_image_proposals, f)


    return all_image_proposals


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    # Initialize the dataset
    potholes_dataset = Potholes(split='Train', folder_path='Potholes', transform=transform)



    # Process 10 images from the dataset
    num_images = 1
    iou_threshold = 0
    # Generate proposals for the entire dataset
    all_image_proposals = generate_proposals_for_entire_dataset(iou_threshold, num_images, pickle=False)

    # get the image from the all_image_proposals dictionary 
    image = all_image_proposals[0]['image']
    image_proposals = all_image_proposals[0]['proposals']
    target = all_image_proposals[0]['ground_truhs']


# Visualize the proposals
visualize_proposals_and_gt(image, target, image_proposals, iou_threshold)




#### Visualize the proposals #### 
#if __name__ == "__main__":
#    transform = transforms.Compose([
#        #transforms.Resize((256, 256)),
#        transforms.ToTensor(),
#    ])
#    # Initialize the dataset
#    potholes_dataset = Potholes(split='Train', folder_path='Potholes', transform=transform)
#
#        # Load image and ground truth boxes
#
#    # Define IoU thresholds
#    iou_thresholds = np.arange(0.0, 1.05, 0.01)  # IoU thresholds from 0.1 to 1.0 with a step of 0.05
#
#    # Process 10 images from the dataset
## Process 10 images from the dataset
#num_images = 10
## Initialize the plot
#fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#
#for i in range(num_images):
#    # Load image and ground truth boxes
#    image, target = potholes_dataset[i + np.random.randint(0, 500)]
#    ground_truth_count = len(target)  # Assuming target is a list or dict of ground truths
#
#    # Generate proposals using selective search
#    proposals = generate_proposals_selective_search(image, max_proposals=9000, type='quality')
#    print(f"Number of proposals for image {i+1}: {len(proposals)}")
#
#    # Lists to store recall and number of proposals for each threshold for the current image
#    recalls = []
#    number_of_proposals = []
#
#    for threshold in iou_thresholds:
#        recall, matches, all_matching_proposals = calculate_recall(proposals, target, threshold)
#        recalls.append(recall)
#        number_of_proposals.append(len(all_matching_proposals))
#
#        # Optional: Print matches and matching proposals for debugging purposes
#        print(f"\nIoU Threshold: {threshold:.2f}")
#        print(f"Recall: {recall:.4f}")
#        print("Matches:")
#        for gt_idx, matched_props in matches.items():
#            print(f"  Ground truth box {gt_idx}:")
#            for match in matched_props:
#                print(f"    Proposal {match['proposal_idx']} - IoU: {match['iou']:.4f} - BBox: {match['proposal_bbox']}")
#        print(f"Number of unique matching proposals: {len(all_matching_proposals)}")
#            
#        # Plot Number of Proposals vs IoU Threshold for the current image on the second subplot
#    axes[1].plot(
#            iou_thresholds,
#            number_of_proposals,
#            marker='o',
#            label=f'Image {i+1} (Ground truths: {ground_truth_count})'
#        )
#        
#        # Plot Recall vs IoU Threshold for the current image on the first subplot
#    axes[0].plot(
#            iou_thresholds,
#            recalls,
#            marker='o',
#            label=f'Image {i+1} (Ground truths: {ground_truth_count})'
#        )
#
## Add titles, labels, and legends to the subplots
#axes[0].set_title("Recall vs IoU Threshold for Each Image")
#
#axes[0].set_xlabel("IoU Threshold")
#axes[0].set_ylabel("Recall")
#axes[0].legend(loc='best')
#axes[0].grid(True)
#
#axes[1].set_title("Number of Matching Proposals vs IoU Threshold for Each Image")
#axes[1].set_xlabel("IoU Threshold")
#axes[1].set_yscale('log')  # Set the y-axis to logarithmic scale
#axes[1].set_ylabel("Number of Matching Proposals")
#axes[1].legend(loc='best')
#axes[1].grid(True)
#
## Adjust layout and show the plot
#plt.tight_layout()
#plt.savefig('recall_proposals.svg')
#plt.show()