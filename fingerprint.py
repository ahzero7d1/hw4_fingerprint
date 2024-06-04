import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filter(img, kernel):
    # Apply the filter
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

def apply_morphology(img, kernel_size):
    # Morphological operations to remove noise
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=0)
    
    # Change pixel values exceeding 200 to 255
    img[img > 200] = 255
    img[img < 50] = 0
    
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=0)
    return img


def get_fp_feature(origin_img, flg_show=False, morph_kernel_size=3):
    # Apply morphology
    img = origin_img
    
    # Apply filtering
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    filtered_img = apply_filter(img, kernel)

    filtered_img = apply_morphology(filtered_img, morph_kernel_size)
    
    # Adaptive thresholding
    binarized = cv2.adaptiveThreshold(filtered_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Skeletonize the image
    skeleton = cv2.ximgproc.thinning(binarized)

    # Padding to avoid border issues
    padded_skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0)

    # Function to find minutiae using 3x3 convolution
    def find_minutiae(img):
        minutiae_end = []
        minutiae_bif = []
        rows, cols = img.shape
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                block = img[i-1:i+2, j-1:j+2]
                P1, P2, P3, P4, P5, P6, P7, P8, P9 = block.flatten()

                if P5 == 255:
                    # Calculate the number of 0->1 transitions in the ordered sequence
                    neighbors = [P2, P3, P6, P9, P8, P7, P4, P1, P2]
                    transitions = sum((neighbors[k] == 0 and neighbors[k+1] == 255) for k in range(8))

                    if transitions == 1:
                        if i != 1 and i != 256:  # Exclude x-coordinates 1 and 256
                            minutiae_end.append((i, j))
                    elif transitions == 3:
                        minutiae_bif.append((i, j))

                    
        return minutiae_end, minutiae_bif


    # Find minutiae
    minutiae_end, minutiae_bif = find_minutiae(padded_skeleton)
    
    if flg_show:
        # Display results
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        axs[0, 0].imshow(origin_img, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(filtered_img, cmap='gray')
        axs[0, 1].set_title('Filtered Image')
        axs[0, 2].imshow(binarized, cmap='gray')
        axs[0, 2].set_title('Binarized Image')
        
        # Mark minutiae on images
        img_minutiae = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        for x, y in minutiae_end:
            cv2.drawMarker(img_minutiae, (y, x), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=3)  # Blue for endings
        for x, y in minutiae_bif:
            cv2.drawMarker(img_minutiae, (y, x), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=3)  # Red for bifurcations
        
        axs[1, 0].imshow(img_minutiae)
        axs[1, 0].set_title('Minutiae Marked')

        # Display skeleton and combined edges
        axs[1, 1].imshow(skeleton, cmap='gray')
        axs[1, 1].set_title('Skeleton')
        
        for ax in axs.flat:
            ax.axis('off')
        
        plt.show()
    
    return minutiae_end, minutiae_bif



import numpy as np
import matplotlib.pyplot as plt

def match_finger(feat_query, feat_train, threshold, draw=False, img_query=None, img_train=None):
    dists = []
    len_match = 0
    
    match_lines = []  # Store the lines connecting matched points
    
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(feat_query))))  # Generate a set of colors
    
    for query_point in feat_query:
        min_dist = float('inf')
        matched_point = None

        for train_point in feat_train:
            dist = np.linalg.norm(np.array(query_point) - np.array(train_point))
            if dist < min_dist:
                min_dist = dist
                matched_point = train_point
                
        if min_dist <= threshold:
            len_match += 1
            dists.append(min_dist)
            match_lines.append((query_point, matched_point))
    
    # Calculate the overall distance between matched points
    overall_dist = np.mean(dists)
    
    if draw and img_query is not None and img_train is not None:
        plt.figure()
        plt.imshow(np.hstack((img_query, img_train)), cmap='gray')
        for line, color in zip(match_lines, colors):
            plt.plot([line[0][1], line[1][1] + img_query.shape[1]], [line[0][0], line[1][0]], color=color, linewidth=0.5, linestyle='-')  # connection line (color determined by iterated color list, thinner line)
            plt.plot(line[0][1], line[0][0], marker='o', markersize=3, color=color)  # query point (color determined by iterated color list)
            plt.plot(line[1][1] + img_query.shape[1], line[1][0], marker='o', markersize=3, color=color)  # matched point in train image (color determined by iterated color list)
        plt.show()

    return overall_dist, overall_dist, len_match
