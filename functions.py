import cv2
import numpy as np
import scipy.io

def calc_ssim_components(img1, img2):
    """Calculate SSIM components separately (luminance, contrast, structure)"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # Separate components
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # Return mean values
    return luminance, contrast_structure


def calc_ssim(img1, img2):
    """Original SSIM calculation"""
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)


def calc_msssim(img1, img2, scales=5):
    """
    Calculate MS-SSIM (Multi-Scale SSIM) - FIXED VERSION
    
    Args:
        img1, img2: Input images (BGR format)
        scales: Number of scales (default: 5)
    
    Returns:
        MS-SSIM score
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Weights for each scale (from original MS-SSIM paper)
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    
    # Check if image is large enough for all scales
    min_size = min(gray1.shape[0], gray1.shape[1])
    max_possible_scales = 0
    temp_size = min_size
    while temp_size >= 11 and max_possible_scales < scales:
        max_possible_scales += 1
        temp_size //= 2
    
    actual_scales = min(scales, max_possible_scales)
    
    if actual_scales < 2:
        # Image too small for MS-SSIM, fall back to regular SSIM
        return calc_ssim(img1, img2)
    
    # Use only the available number of scales
    weights = weights[:actual_scales]
    weights = weights / weights.sum()  # Normalize
    
    # Lists to store contrast-structure values
    mcs_list = []
    
    current_img1 = gray1.copy()
    current_img2 = gray2.copy()
    
    # Calculate CS at each scale
    for i in range(actual_scales):
        luminance, cs = calc_ssim_components(current_img1, current_img2)
        
        if i == actual_scales - 1:
            # Last scale: save both luminance and cs
            final_luminance = np.mean(luminance)
            mcs_list.append(np.mean(cs))
        else:
            # Other scales: save only cs
            mcs_list.append(np.mean(cs))
            
        # Downsample for next scale
        if i < actual_scales - 1:
            current_img1 = cv2.pyrDown(current_img1)
            current_img2 = cv2.pyrDown(current_img2)
    
    # Calculate MS-SSIM using the formula:
    # MS-SSIM = [luminance_M]^weight_M * Product(cs_j^weight_j) for j=1 to M
    msssim = np.prod([mcs_list[i] ** weights[i] for i in range(actual_scales - 1)])
    msssim *= (final_luminance ** weights[actual_scales - 1]) * (mcs_list[actual_scales - 1] ** weights[actual_scales - 1])
    
    return msssim

def calculate_prcc(ssim_array, dmos_array):
    # Convert to numpy arrays
    ssim_array = np.array(ssim_array, dtype=float)
    dmos_array = np.array(dmos_array, dtype=float)
    
    # Check if arrays have the same length
    if len(ssim_array) != len(dmos_array):
        raise ValueError("SSIM and DMOS arrays must have the same length")
    
    if len(ssim_array) < 2:
        raise ValueError("Arrays must contain at least 2 samples")
    
    # Calculate means
    mean_ssim = np.mean(ssim_array)
    mean_dmos = np.mean(dmos_array)
    
    # Deviations from mean
    ssim_dev = ssim_array - mean_ssim
    dmos_dev = dmos_array - mean_dmos
    
    # Covariance (numerator)
    covariance = np.sum(ssim_dev * dmos_dev)
    
    # Standard deviations (denominator)
    ssim_std = np.sqrt(np.sum(ssim_dev ** 2))
    dmos_std = np.sqrt(np.sum(dmos_dev ** 2))
    
    # Pearson correlation
    prcc = covariance / (ssim_std * dmos_std)
    
    return prcc


def calculate_sroc(ssim_array, dmos_array):
    # Convert to numpy arrays
    ssim_array = np.array(ssim_array, dtype=float)
    dmos_array = np.array(dmos_array, dtype=float)
    
    # Check if arrays have the same length
    if len(ssim_array) != len(dmos_array):
        raise ValueError("SSIM and DMOS arrays must have the same length")
    
    if len(ssim_array) < 2:
        raise ValueError("Arrays must contain at least 2 samples")
    
    # Convert values to ranks
    ssim_ranks = rankdata(ssim_array)
    dmos_ranks = rankdata(dmos_array)
    
    # Apply Pearson formula to ranks
    mean_ssim_rank = np.mean(ssim_ranks)
    mean_dmos_rank = np.mean(dmos_ranks)
    
    ssim_rank_dev = ssim_ranks - mean_ssim_rank
    dmos_rank_dev = dmos_ranks - mean_dmos_rank
    
    rank_covariance = np.sum(ssim_rank_dev * dmos_rank_dev)
    ssim_rank_std = np.sqrt(np.sum(ssim_rank_dev ** 2))
    dmos_rank_std = np.sqrt(np.sum(dmos_rank_dev ** 2))
    
    sroc = rank_covariance / (ssim_rank_std * dmos_rank_std)
    
    return sroc


def calculate_rmse(predicted, actual):
    # Convert to numpy arrays
    predicted = np.array(predicted, dtype=float)
    actual = np.array(actual, dtype=float)
    
    # Check if arrays have the same length
    if len(predicted) != len(actual):
        raise ValueError("Predicted and actual arrays must have the same length")
    
    if len(predicted) < 1:
        raise ValueError("Arrays must contain at least 1 sample")
    
    # Calculate squared differences
    squared_errors = (predicted - actual) ** 2
    
    # Calculate mean of squared errors
    mean_squared_error = np.mean(squared_errors)
    
    # Take square root
    rmse = np.sqrt(mean_squared_error)
    
    return rmse


def rankdata(data):
    n = len(data)
    # Create array of (value, original_index) pairs
    indexed_data = [(data[i], i) for i in range(n)]
    
    # Sort by value
    indexed_data.sort(key=lambda x: x[0])
    
    # Assign ranks
    ranks = np.zeros(n)
    i = 0
    while i < n:
        # Find ties
        j = i
        while j < n and indexed_data[j][0] == indexed_data[i][0]:
            j += 1
        
        # Assign average rank to all tied values
        avg_rank = (i + j + 1) / 2  # +1 because ranks start at 1
        for k in range(i, j):
            original_idx = indexed_data[k][1]
            ranks[original_idx] = avg_rank
        
        i = j
    
    return ranks
import pandas as pd
def extract_live_dmos(filename):
    """
    Extract DMOS values from LIVE database Excel file.
    The LIVE database often stores DMOS as comma-separated values.
    
    Parameters:
    -----------
    filename : str
        Path to the Excel file containing LIVE database data
    
    Returns:
    --------
    dmos_array : numpy array
        Array of DMOS values as floats
    """
    # Read the Excel file
    data = pd.read_excel(filename, header=None)
    
    # Get the first cell (which contains all the comma-separated values)
    dmos_string = str(data.iloc[0, 0])
    
    # Split by comma and convert to float
    dmos_values = dmos_string.split(',')
    dmos_array = np.array([float(val) for val in dmos_values if val.strip()])
    
    print("DMOS values extracted successfully!")
    print(f"\nTotal number of images: {len(dmos_array)}")
    print(f"\nDMOS statistics:")
    print(f"  Min: {np.min(dmos_array):.3f}")
    print(f"  Max: {np.max(dmos_array):.3f}")
    print(f"  Mean: {np.mean(dmos_array):.3f}")
    print(f"  Std: {np.std(dmos_array):.3f}")
    print(f"\nFirst 10 DMOS values:")
    print(dmos_array[:10])
    print(f"\nLast 10 DMOS values:")
    print(dmos_array[-10:])
    
    return dmos_array

def file_paths():
    file_images= [
        'fastfading',
        'gblur',
        'jp2k',
        'jpeg',
        'wn',
    ]
    
    
    images = {}
    
    for f in file_images:
        with open(f"C:\\Users\\Hassiba Informatique\\Desktop\\Masters Degree\\M2\\QDM\\dataset\\{f}\\info.txt", 'r') as file:
            for line in file:
                if line != '\n':
                    parts = line.strip().split(" ")
                    if parts[0] not in images.keys():
                        images[parts[0]] = {f : parts[1:]}  
                    else:
                        if f in images[parts[0]].keys():
                            images[parts[0]][f] += parts[1:]
                        else:
                            images[parts[0]][f] = parts[1:]
    
    return images

def dmos_results():
    file_names = [
            'jp2k',
            'jpeg',
            'wn',
            'gblur',
            'fastfading',
        ]
    mat_file_path = "C:\\Users\\Hassiba Informatique\\Desktop\\Masters Degree\\M2\\QDM\\dataset\\dmos.mat"
    mat_data = scipy.io.loadmat(mat_file_path)

    dmos = []
    dmos_results = {}
    i = 0

    for key in mat_data.keys():
        if not key.startswith('__'):
            if isinstance(mat_data[key], np.ndarray):
                dmos = mat_data[key][0]
    for f in file_names:
        with open("C:\\Users\\Hassiba Informatique\\Desktop\\Masters Degree\\M2\\QDM\\dataset\\"+f+"\\info.txt", 'r') as source_file:
            for line in source_file:
                if line != "\n":
                    l = line.split(" ")
                    if l[0] not in dmos_results.keys():
                        dmos_results[l[0]] = { f: [{ l[1] : float(dmos[i]) }]}
                    else:
                        if f in dmos_results[l[0]].keys():
                            dmos_results[l[0]][f].append({ l[1]: float(dmos[i]) })
                        else:
                            dmos_results[l[0]][f] = [{ l[1]: float(dmos[i]) }]
                    i += 1
    return dmos_results

# Example usage
if __name__ == "__main__":
    # Sample data
    ssim = [0.95, 0.88, 0.76, 0.82, 0.91, 0.68, 0.79, 0.85]
    dmos = [10, 25, 45, 35, 15, 60, 40, 30]
    
    # Calculate correlations
    prcc = calculate_prcc(ssim, dmos)
    sroc = calculate_sroc(ssim, dmos)
    rmse = calculate_rmse(ssim, dmos)
    
    print("Correlation Results:")
    print(f"PRCC: {prcc:.4f}")
    print(f"SROC: {sroc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if abs(prcc) > 0.9:
        print("- Very strong linear correlation")
    elif abs(prcc) > 0.7:
        print("- Strong linear correlation")
    elif abs(prcc) > 0.5:
        print("- Moderate linear correlation")
    else:
        print("- Weak linear correlation")
        
    filename = "c:\\Users\\Hassiba Informatique\\Desktop\\Masters Degree\\M2\\QDM\\dataset\\dmos.xlsx"
    dmos_data = extract_live_dmos(filename)
    
    print(f"\nFull DMOS array shape: {dmos_data.shape}")
    print(f"Full DMOS array:\n{dmos_data}")