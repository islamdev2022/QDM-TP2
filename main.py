import cv2
import numpy as np
import scipy.io
import os
import tkinter as tk
from tkinter import filedialog
from functions import file_paths, calculate_prcc, calculate_sroc, calculate_rmse, calc_ssim, calc_msssim


def calculate_all_metrics(metric_function, metric_name):
    """
    Calculate a metric for all reference images with their degraded versions.
    Returns a dictionary with metric values organized by degradation type.
    
    Args:
        metric_function: Function to calculate the metric (calc_ssim or calc_msssim)
        metric_name: Name of the metric for display purposes
    """
    # Get the file paths structure
    images_dict = file_paths()
    
    # Base paths
    ref_path = r"C:\Users\Hassiba Informatique\Desktop\Masters Degree\M2\QDM\dataset\refimgs"
    dataset_path = r"C:\Users\Hassiba Informatique\Desktop\Masters Degree\M2\QDM\dataset"
    
    # Initialize arrays for each degradation type
    metric_fastfading = []
    metric_gblur = []
    metric_jp2k = []
    metric_jpeg = []
    metric_wn = []
    
    # Dictionary to map degradation types to their arrays
    metric_arrays = {
        'fastfading': metric_fastfading,
        'gblur': metric_gblur,
        'jp2k': metric_jp2k,
        'jpeg': metric_jpeg,
        'wn': metric_wn
    }
    
    print(f"\n{'='*60}")
    print(f"CALCULATING {metric_name}")
    print(f"{'='*60}")
    
    # Process each reference image
    for ref_image_name, degradations in images_dict.items():
        print(f"\nProcessing reference image: {ref_image_name}")
        
        # Load reference image
        ref_image_path = f"{ref_path}\\{ref_image_name}"
        ref_img = cv2.imread(ref_image_path)
        
        if ref_img is None:
            print(f"  Warning: Could not load reference image {ref_image_path}")
            continue
        
        # Process each degradation type for this reference image
        for deg_type, deg_images_list in degradations.items():
            # deg_images_list contains: [img_name1, value1, img_name2, value2, ...]
            # We need to extract only the image names (every other element starting from index 0)
            deg_image_names = [deg_images_list[i] for i in range(0, len(deg_images_list), 2)]
            
            print(f"  Processing {deg_type}: {len(deg_image_names)} images")
            
            # Calculate metric for each degraded image
            for deg_img_name in deg_image_names:
                deg_image_path = f"{dataset_path}\\{deg_type}\\{deg_img_name}"
                deg_img = cv2.imread(deg_image_path)
                
                if deg_img is None:
                    print(f"    Warning: Could not load {deg_image_path}")
                    continue
                
                # Resize degraded image to match reference image dimensions if needed
                if deg_img.shape != ref_img.shape:
                    deg_img = cv2.resize(deg_img, (ref_img.shape[1], ref_img.shape[0]))
                
                # Calculate metric
                metric_value = metric_function(ref_img, deg_img)
                metric_arrays[deg_type].append(metric_value)
                
                print(f"    {deg_img_name}: {metric_name} = {metric_value:.4f}")
    
    # Create global array containing all metric values in the order: jp2k, jpeg, wn, gblur, fastfading
    global_metric_array = []
    for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
        global_metric_array.extend(metric_arrays[deg_type])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY OF {metric_name} CALCULATIONS")
    print(f"{'='*60}")
    for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
        arr = metric_arrays[deg_type]
        print(f"\n{deg_type.upper()}:")
        print(f"  Total images: {len(arr)}")
        if len(arr) > 0:
            print(f"  {metric_name} - Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f}, Mean: {np.mean(arr):.4f}")
            print(f"  First 5 values: {[f'{v:.4f}' for v in arr[:5]]}")
    
    print(f"\nGLOBAL ARRAY:")
    print(f"  Total {metric_name} values: {len(global_metric_array)}")
    if len(global_metric_array) > 0:
        print(f"  {metric_name} - Min: {np.min(global_metric_array):.4f}, Max: {np.max(global_metric_array):.4f}, Mean: {np.mean(global_metric_array):.4f}")
    
    return {
        'fastfading': metric_fastfading,
        'gblur': metric_gblur,
        'jp2k': metric_jp2k,
        'jpeg': metric_jpeg,
        'wn': metric_wn,
        'global': global_metric_array
    }


def extract_dmos_arrays(metric_results):
    """
    Extract DMOS arrays by slicing the global DMOS array based on metric array lengths.
    The order in dmos.mat matches the order: jp2k, jpeg, wn, gblur, fastfading
    """
    # Load DMOS from mat file
    mat_file_path = r"C:\Users\Hassiba Informatique\Desktop\Masters Degree\M2\QDM\dataset\dmos.mat"
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Extract the DMOS array
    dmos_global = None
    for key in mat_data.keys():
        if not key.startswith('__'):
            if isinstance(mat_data[key], np.ndarray):
                dmos_global = mat_data[key][0]
                break
    
    if dmos_global is None:
        raise ValueError("Could not find DMOS data in mat file")
    
    print(f"\nLoaded global DMOS array with {len(dmos_global)} values")
    
    # The order in the dmos.mat file is: jp2k, jpeg, wn, gblur, fastfading
    # We need to slice based on the metric array lengths
    dmos_arrays = {
        'jp2k': [],
        'jpeg': [],
        'wn': [],
        'gblur': [],
        'fastfading': [],
        'global': dmos_global.tolist()
    }
    
    # Get the counts for each degradation type
    counts = {
        'jp2k': len(metric_results['jp2k']),
        'jpeg': len(metric_results['jpeg']),
        'wn': len(metric_results['wn']),
        'gblur': len(metric_results['gblur']),
        'fastfading': len(metric_results['fastfading'])
    }
    
    print("\nMetric array lengths:")
    for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
        print(f"  {deg_type}: {counts[deg_type]}")
    
    # Slice the global DMOS array according to the order in the mat file
    start_idx = 0
    for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
        count = counts[deg_type]
        dmos_arrays[deg_type] = dmos_global[start_idx:start_idx + count].tolist()
        start_idx += count
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF DMOS EXTRACTION")
    print("="*60)
    for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
        arr = dmos_arrays[deg_type]
        print(f"\n{deg_type.upper()}:")
        print(f"  Total DMOS values: {len(arr)}")
        if len(arr) > 0:
            print(f"  DMOS - Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f}, Mean: {np.mean(arr):.4f}")
            print(f"  First 5 values: {[f'{v:.4f}' for v in arr[:5]]}")
    
    print(f"\nGLOBAL DMOS ARRAY:")
    print(f"  Total DMOS values: {len(dmos_arrays['global'])}")
    print(f"  DMOS - Min: {np.min(dmos_arrays['global']):.4f}, Max: {np.max(dmos_arrays['global']):.4f}, Mean: {np.mean(dmos_arrays['global']):.4f}")
    
    return dmos_arrays


def calculate_performance_metrics(metric_results, dmos_results, metric_name):
    """
    Calculate PRCC, SROC, and RMSE for all degradation types and global.
    """
    print("\n" + "="*60)
    print(f"CALCULATING PERFORMANCE METRICS FOR {metric_name}")
    print("="*60)
    
    metrics = {}
    
    # List of degradation types
    deg_types = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading', 'global']
    
    for deg_type in deg_types:
        metric_array = metric_results[deg_type]
        dmos_array = dmos_results[deg_type]
        
        print(f"\n{deg_type.upper()}:")
        print(f"  {metric_name} array length: {len(metric_array)}")
        print(f"  DMOS array length: {len(dmos_array)}")
        
        if len(metric_array) != len(dmos_array):
            print(f"  ERROR: Array length mismatch!")
            metrics[deg_type] = {'prcc': None, 'sroc': None, 'rmse': None}
            continue
        
        if len(metric_array) < 2:
            print(f"  ERROR: Not enough data points!")
            metrics[deg_type] = {'prcc': None, 'sroc': None, 'rmse': None}
            continue
        
        # Calculate metrics
        try:
            prcc = calculate_prcc(metric_array, dmos_array)
            sroc = calculate_sroc(metric_array, dmos_array)
            rmse = calculate_rmse(metric_array, dmos_array)
            
            metrics[deg_type] = {
                'prcc': prcc,
                'sroc': sroc,
                'rmse': rmse
            }
            
            print(f"  PRCC: {prcc:.4f}")
            print(f"  SROC: {sroc:.4f}")
            print(f"  RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"  ERROR calculating metrics: {e}")
            metrics[deg_type] = {'prcc': None, 'sroc': None, 'rmse': None}
    
    return metrics


def print_metrics_table(metrics, metric_name):
    """
    Print a summary table of all metrics.
    """
    print("\n" + "="*60)
    print(f"{metric_name} PERFORMANCE METRICS SUMMARY")
    print("="*60)
    print(f"\n{'Degradation Type':<15} {'PRCC':<10} {'SROC':<10} {'RMSE':<10}")
    print("-" * 60)
    
    for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading', 'global']:
        if deg_type in metrics and metrics[deg_type]['prcc'] is not None:
            prcc = metrics[deg_type]['prcc']
            sroc = metrics[deg_type]['sroc']
            rmse = metrics[deg_type]['rmse']
            print(f"{deg_type.upper():<15} {prcc:<10.4f} {sroc:<10.4f} {rmse:<10.4f}")
        else:
            print(f"{deg_type.upper():<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("="*60)


def compare_metrics(ssim_metrics, msssim_metrics):
    """
    Compare SSIM and MS-SSIM performance metrics side by side.
    """
    print("\n" + "="*70)
    print("COMPARISON: SSIM vs MS-SSIM")
    print("="*70)
    
    deg_types = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading', 'global']
    
    # PRCC Comparison
    print("\n" + "-"*70)
    print("PRCC (Pearson Correlation) - Higher is Better")
    print("-"*70)
    print(f"{'Degradation':<15} {'SSIM':<12} {'MS-SSIM':<12} {'Improvement':<15} {'Winner':<10}")
    print("-"*70)
    
    prcc_improvements = []
    for deg_type in deg_types:
        if (deg_type in ssim_metrics and ssim_metrics[deg_type]['prcc'] is not None and
            deg_type in msssim_metrics and msssim_metrics[deg_type]['prcc'] is not None):
            
            ssim_prcc = ssim_metrics[deg_type]['prcc']
            msssim_prcc = msssim_metrics[deg_type]['prcc']
            improvement = msssim_prcc - ssim_prcc
            winner = "MS-SSIM" if msssim_prcc > ssim_prcc else "SSIM" if ssim_prcc > msssim_prcc else "TIE"
            
            prcc_improvements.append(improvement)
            
            print(f"{deg_type.upper():<15} {ssim_prcc:<12.4f} {msssim_prcc:<12.4f} {improvement:+<15.4f} {winner:<10}")
        else:
            print(f"{deg_type.upper():<15} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
    
    # SROC Comparison
    print("\n" + "-"*70)
    print("SROC (Spearman Correlation) - Higher is Better")
    print("-"*70)
    print(f"{'Degradation':<15} {'SSIM':<12} {'MS-SSIM':<12} {'Improvement':<15} {'Winner':<10}")
    print("-"*70)
    
    sroc_improvements = []
    for deg_type in deg_types:
        if (deg_type in ssim_metrics and ssim_metrics[deg_type]['sroc'] is not None and
            deg_type in msssim_metrics and msssim_metrics[deg_type]['sroc'] is not None):
            
            ssim_sroc = ssim_metrics[deg_type]['sroc']
            msssim_sroc = msssim_metrics[deg_type]['sroc']
            improvement = msssim_sroc - ssim_sroc
            winner = "MS-SSIM" if msssim_sroc > ssim_sroc else "SSIM" if ssim_sroc > msssim_sroc else "TIE"
            
            sroc_improvements.append(improvement)
            
            print(f"{deg_type.upper():<15} {ssim_sroc:<12.4f} {msssim_sroc:<12.4f} {improvement:+<15.4f} {winner:<10}")
        else:
            print(f"{deg_type.upper():<15} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
    
    # RMSE Comparison
    print("\n" + "-"*70)
    print("RMSE (Root Mean Square Error) - Lower is Better")
    print("-"*70)
    print(f"{'Degradation':<15} {'SSIM':<12} {'MS-SSIM':<12} {'Improvement':<15} {'Winner':<10}")
    print("-"*70)
    
    rmse_improvements = []
    for deg_type in deg_types:
        if (deg_type in ssim_metrics and ssim_metrics[deg_type]['rmse'] is not None and
            deg_type in msssim_metrics and msssim_metrics[deg_type]['rmse'] is not None):
            
            ssim_rmse = ssim_metrics[deg_type]['rmse']
            msssim_rmse = msssim_metrics[deg_type]['rmse']
            improvement = ssim_rmse - msssim_rmse  # Positive means MS-SSIM is better
            winner = "MS-SSIM" if msssim_rmse < ssim_rmse else "SSIM" if ssim_rmse < msssim_rmse else "TIE"
            
            rmse_improvements.append(improvement)
            
            print(f"{deg_type.upper():<15} {ssim_rmse:<12.4f} {msssim_rmse:<12.4f} {improvement:+<15.4f} {winner:<10}")
        else:
            print(f"{deg_type.upper():<15} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
    
    # Overall Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    if prcc_improvements:
        avg_prcc_improvement = np.mean(prcc_improvements)
        print(f"\nAverage PRCC Improvement: {avg_prcc_improvement:+.4f}")
        print(f"MS-SSIM wins on PRCC: {sum(1 for x in prcc_improvements if x > 0)}/{len(prcc_improvements)} cases")
    
    if sroc_improvements:
        avg_sroc_improvement = np.mean(sroc_improvements)
        print(f"\nAverage SROC Improvement: {avg_sroc_improvement:+.4f}")
        print(f"MS-SSIM wins on SROC: {sum(1 for x in sroc_improvements if x > 0)}/{len(sroc_improvements)} cases")
    
    if rmse_improvements:
        avg_rmse_improvement = np.mean(rmse_improvements)
        print(f"\nAverage RMSE Improvement: {avg_rmse_improvement:+.4f} (positive = MS-SSIM better)")
        print(f"MS-SSIM wins on RMSE: {sum(1 for x in rmse_improvements if x > 0)}/{len(rmse_improvements)} cases")
    
    # Final Verdict
    print("\n" + "="*70)
    if prcc_improvements and sroc_improvements and rmse_improvements:
        msssim_wins = (sum(1 for x in prcc_improvements if x > 0) + 
                       sum(1 for x in sroc_improvements if x > 0) + 
                       sum(1 for x in rmse_improvements if x > 0))
        total_comparisons = len(prcc_improvements) + len(sroc_improvements) + len(rmse_improvements)
        
        print(f"FINAL VERDICT:")
        print(f"MS-SSIM outperforms SSIM in {msssim_wins}/{total_comparisons} comparisons")
        print(f"Win rate: {(msssim_wins/total_comparisons)*100:.1f}%")
        
        if msssim_wins > total_comparisons / 2:
            print("\n[+] MS-SSIM shows BETTER performance overall!")
        elif msssim_wins < total_comparisons / 2:
            print("\n[-] SSIM shows BETTER performance overall!")
            print("\nPossible reasons MS-SSIM performed poorly:")
            print("  - Images may be too small for effective multi-scale analysis")
            print("  - Dataset characteristics may not benefit from multi-scale approach")
            print("  - Single-scale features may be more relevant for these degradation types")
        else:
            print("\n[=] Both metrics perform EQUALLY!")
    
    print("="*70)


def interactive_mode():
    """
    Interactive mode: Let user choose a reference image and a degraded image using file explorer.
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE - SINGLE IMAGE COMPARISON")
    print("="*70)
    
    ref_path = r"C:\Users\Hassiba Informatique\Desktop\Masters Degree\M2\QDM\dataset\refimgs"
    dataset_path = r"C:\Users\Hassiba Informatique\Desktop\Masters Degree\M2\QDM\dataset"
    
    # Create a hidden root window for file dialogs
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Step 1: Choose reference image
    print("\n" + "-"*70)
    print("STEP 1: Choose a Reference Image")
    print("-"*70)
    print("\nOpening file explorer for reference images...")
    print(f"Location: {ref_path}")
    
    ref_image_path = filedialog.askopenfilename(
        initialdir=ref_path,
        title="Select Reference Image",
        filetypes=[
            ("Image files", "*.bmp *.jpg *.jpeg *.png"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not ref_image_path:
        print("\n[X] No reference image selected. Exiting...")
        root.destroy()
        return
    
    selected_ref = os.path.basename(ref_image_path)
    print(f"\n[✓] Selected reference image: {selected_ref}")
    
    # Load reference image
    ref_img = cv2.imread(ref_image_path)
    
    if ref_img is None:
        print(f"ERROR: Could not load reference image {ref_image_path}")
        root.destroy()
        return
    
    print(f"    Image loaded successfully! Dimensions: {ref_img.shape[1]}x{ref_img.shape[0]}")
    
    # Step 2: Get degraded images for this reference image
    print("\n" + "-"*70)
    print("STEP 2: Choose a Degraded Image")
    print("-"*70)
    
    images_dict = file_paths()
    
    if selected_ref not in images_dict:
        print(f"\n[!] Warning: No degraded versions found in database for {selected_ref}")
        print("    You can still select any degraded image manually.")
        
        print("\nOpening file explorer for degraded images...")
        print(f"Location: {dataset_path}")
        
        deg_image_path = filedialog.askopenfilename(
            initialdir=dataset_path,
            title="Select Degraded Image",
            filetypes=[
                ("Image files", "*.bmp *.jpg *.jpeg *.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not deg_image_path:
            print("\n[X] No degraded image selected. Exiting...")
            root.destroy()
            return
        
        selected_deg_name = os.path.basename(deg_image_path)
        # Try to detect degradation type from path
        deg_type = "unknown"
        for dtype in ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']:
            if dtype in deg_image_path.lower():
                deg_type = dtype
                break
        
        selected_deg = {
            'name': selected_deg_name,
            'type': deg_type,
            'path': deg_image_path
        }
    else:
        # Collect all degraded images for this reference
        degraded_images = []
        available_types = []
        valid_image_names = []
        
        for deg_type, deg_images_list in images_dict[selected_ref].items():
            # Extract image names (every other element)
            deg_image_names = [deg_images_list[i] for i in range(0, len(deg_images_list), 2)]
            available_types.append(deg_type)
            valid_image_names.extend(deg_image_names)
            for deg_img_name in deg_image_names:
                degraded_images.append({
                    'name': deg_img_name,
                    'type': deg_type,
                    'path': os.path.join(dataset_path, deg_type, deg_img_name)
                })
        
        print(f"\nFound {len(degraded_images)} degraded versions of {selected_ref}")
        print(f"Available degradation types: {', '.join(available_types)}")
        
        # Create a temporary directory with symbolic links to only the valid images
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp(prefix="filtered_images_")
        
        try:
            # Copy only the relevant degraded images to temp directory
            for deg_info in degraded_images:
                if os.path.exists(deg_info['path']):
                    dest_folder = os.path.join(temp_dir, deg_info['type'])
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_path = os.path.join(dest_folder, deg_info['name'])
                    shutil.copy2(deg_info['path'], dest_path)
            
            print(f"\nShowing only the {len(degraded_images)} images related to {selected_ref}")
            print("\nOpening file explorer with filtered images...")
            print(f"Navigate through the degradation type folders to see the images.")
            
            deg_image_path = filedialog.askopenfilename(
                initialdir=temp_dir,
                title=f"Select Degraded Version of {selected_ref}",
                filetypes=[
                    ("Image files", "*.bmp *.jpg *.jpeg *.png"),
                    ("BMP files", "*.bmp"),
                    ("All files", "*.*")
                ]
            )
            
            if not deg_image_path:
                print("\n[X] No degraded image selected. Exiting...")
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                root.destroy()
                return
            
            selected_deg_name = os.path.basename(deg_image_path)
            
            # Find the actual path and degradation type
            deg_type = "unknown"
            actual_path = deg_image_path
            
            for deg_info in degraded_images:
                if deg_info['name'] == selected_deg_name:
                    deg_type = deg_info['type']
                    actual_path = deg_info['path']
                    break
            
            # If not found in database, try to detect from path
            if deg_type == "unknown":
                for dtype in ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']:
                    if dtype in deg_image_path.lower():
                        deg_type = dtype
                        # Reconstruct actual path
                        actual_path = os.path.join(dataset_path, deg_type, selected_deg_name)
                        break
            
            selected_deg = {
                'name': selected_deg_name,
                'type': deg_type,
                'path': actual_path
            }
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"\n[!] Error creating filtered view: {e}")
            print("    Falling back to normal file selection...")
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            deg_image_path = filedialog.askopenfilename(
                initialdir=dataset_path,
                title=f"Select Degraded Version of {selected_ref}",
                filetypes=[
                    ("Image files", "*.bmp *.jpg *.jpeg *.png"),
                    ("BMP files", "*.bmp"),
                    ("All files", "*.*")
                ]
            )
            
            if not deg_image_path:
                print("\n[X] No degraded image selected. Exiting...")
                root.destroy()
                return
            
            selected_deg_name = os.path.basename(deg_image_path)
            
            # Find the degradation type
            deg_type = "unknown"
            for deg_info in degraded_images:
                if deg_info['name'] == selected_deg_name:
                    deg_type = deg_info['type']
                    break
            
            # If not found in database, try to detect from path
            if deg_type == "unknown":
                for dtype in ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']:
                    if dtype in deg_image_path.lower():
                        deg_type = dtype
                        break
            
            selected_deg = {
                'name': selected_deg_name,
                'type': deg_type,
                'path': deg_image_path
            }
    
    print(f"\n[✓] Selected degraded image: {selected_deg['name']} ({selected_deg['type']})")
    
    # Load degraded image
    deg_img = cv2.imread(selected_deg['path'])
    
    if deg_img is None:
        print(f"ERROR: Could not load degraded image {selected_deg['path']}")
        root.destroy()
        return
    
    print(f"    Image loaded successfully! Dimensions: {deg_img.shape[1]}x{deg_img.shape[0]}")
    
    # Resize if needed
    if deg_img.shape != ref_img.shape:
        print(f"    Resizing degraded image to match reference dimensions...")
        deg_img = cv2.resize(deg_img, (ref_img.shape[1], ref_img.shape[0]))
    
    # Destroy the root window
    root.destroy()
    
    # Step 3: Calculate metrics
    print("\n" + "-"*70)
    print("STEP 3: Calculating Quality Metrics")
    print("-"*70)
    
    print("\nCalculating SSIM...")
    ssim_value = calc_ssim(ref_img, deg_img)
    
    print("Calculating MS-SSIM...")
    msssim_value = calc_msssim(ref_img, deg_img)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nReference Image:  {selected_ref}")
    print(f"Degraded Image:   {selected_deg['name']}")
    print(f"Degradation Type: {selected_deg['type'].upper()}")
    print(f"\n{'Metric':<20} {'Value':<15} {'Range':<20}")
    print("-"*70)
    print(f"{'SSIM':<20} {ssim_value:<15.6f} {'[0.0 - 1.0, higher better]':<20}")
    print(f"{'MS-SSIM':<20} {msssim_value:<15.6f} {'[0.0 - 1.0, higher better]':<20}")
    print("-"*70)
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    def interpret_quality(score):
        if score >= 0.95:
            return "Excellent - Almost imperceptible degradation"
        elif score >= 0.90:
            return "Very Good - Minor degradation"
        elif score >= 0.80:
            return "Good - Noticeable but acceptable degradation"
        elif score >= 0.70:
            return "Fair - Moderate degradation"
        elif score >= 0.50:
            return "Poor - Significant degradation"
        else:
            return "Very Poor - Severe degradation"
    
    print(f"\nSSIM Score ({ssim_value:.4f}):")
    print(f"  {interpret_quality(ssim_value)}")
    
    print(f"\nMS-SSIM Score ({msssim_value:.4f}):")
    print(f"  {interpret_quality(msssim_value)}")
    
    if abs(ssim_value - msssim_value) > 0.05:
        if msssim_value > ssim_value:
            print(f"\n[!] MS-SSIM rates this image {(msssim_value - ssim_value)*100:.2f}% higher than SSIM")
            print("    Multi-scale analysis may be capturing quality aspects better.")
        else:
            print(f"\n[!] SSIM rates this image {(ssim_value - msssim_value)*100:.2f}% higher than MS-SSIM")
            print("    Single-scale features may be more relevant for this degradation.")
    else:
        print("\n[=] Both metrics agree closely on the quality assessment.")
    
    print("\n" + "="*70)


def batch_mode():
    """
    Batch mode: Calculate metrics for all images and compare SSIM vs MS-SSIM.
    """
    print("\n" + "="*70)
    print("BATCH MODE - COMPLETE DATASET ANALYSIS")
    print("="*70)
    
    # Step 1: Calculate SSIM for all images
    print("\n\n" + "#"*70)
    print("# PART 1: CALCULATING BASIC SSIM")
    print("#"*70)
    ssim_results = calculate_all_metrics(calc_ssim, "SSIM")
    
    # Step 2: Calculate MS-SSIM for all images
    print("\n\n" + "#"*70)
    print("# PART 2: CALCULATING MS-SSIM")
    print("#"*70)
    msssim_results = calculate_all_metrics(calc_msssim, "MS-SSIM")
    
    # Step 3: Extract DMOS arrays (only need to do this once)
    print("\n\n" + "#"*70)
    print("# PART 3: EXTRACTING DMOS VALUES")
    print("#"*70)
    dmos_arrays = extract_dmos_arrays(ssim_results)
    
    # Step 4: Calculate performance metrics for SSIM
    print("\n\n" + "#"*70)
    print("# PART 4: CALCULATING SSIM PERFORMANCE METRICS")
    print("#"*70)
    ssim_metrics = calculate_performance_metrics(ssim_results, dmos_arrays, "SSIM")
    print_metrics_table(ssim_metrics, "SSIM")
    
    # Step 5: Calculate performance metrics for MS-SSIM
    print("\n\n" + "#"*70)
    print("# PART 5: CALCULATING MS-SSIM PERFORMANCE METRICS")
    print("#"*70)
    msssim_metrics = calculate_performance_metrics(msssim_results, dmos_arrays, "MS-SSIM")
    print_metrics_table(msssim_metrics, "MS-SSIM")
    
    # Step 6: Compare both metrics
    print("\n\n" + "#"*70)
    print("# PART 6: COMPARISON RESULTS")
    print("#"*70)
    compare_metrics(ssim_metrics, msssim_metrics)
    
    print("\n" + "="*70)
    print("ALL CALCULATIONS AND COMPARISON COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("IMAGE QUALITY ASSESSMENT TOOL")
    print("SSIM vs MS-SSIM Analysis")
    print("="*70)
    
    print("\nSelect Mode:")
    print("  1. Interactive Mode - Compare single images")
    print("  2. Batch Mode - Analyze complete dataset")
    
    while True:
        try:
            mode_choice = input("\nEnter your choice (1 or 2): ").strip()
            if mode_choice in ['1', '2']:
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)
    
    if mode_choice == '1':
        interactive_mode()
    else:
        batch_mode()