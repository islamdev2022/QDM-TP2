import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import scipy.io
import os
import threading
import tempfile
import shutil
from functions import file_paths, calculate_prcc, calculate_sroc, calculate_rmse, calc_ssim, calc_msssim

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ImageQualityAssessmentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Image Quality Assessment Tool - SSIM vs MS-SSIM")
        self.geometry("1400x950")
        
        # Dataset path
        self.dataset_path = ctk.StringVar(value="")
        
        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create header
        self.create_header()
        
        # Create tabview
        self.create_tabview()
        
        # Store selected images
        self.ref_image = None
        self.ref_image_path = None
        self.ref_image_name = None
        self.deg_image = None
        self.deg_image_path = None
        self.deg_type = None
        self.temp_dir = None
    
    def create_header(self):
        """Create the header section with dataset path selection"""
        header_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray15"), corner_radius=15)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Title with gradient effect
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎨 Image Quality Assessment Tool",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#1f538d", "#3a7ebf")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(15, 5))
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Compare SSIM vs MS-SSIM Performance",
            font=ctk.CTkFont(size=14),
            text_color=("gray50", "gray70")
        )
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 15))
        
        # Dataset path section with icon
        path_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        path_container.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=(0, 15))
        path_container.grid_columnconfigure(1, weight=1)
        
        path_icon = ctk.CTkLabel(
            path_container,
            text="📂",
            font=ctk.CTkFont(size=20)
        )
        path_icon.grid(row=0, column=0, padx=(0, 10))
        
        path_label = ctk.CTkLabel(
            path_container,
            text="Dataset Location:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        path_label.grid(row=0, column=1, sticky="w")
        
        self.path_entry = ctk.CTkEntry(
            path_container,
            textvariable=self.dataset_path,
            placeholder_text="Click browse to select your dataset folder...",
            height=40,
            font=ctk.CTkFont(size=13),
            border_width=2
        )
        self.path_entry.grid(row=1, column=0, columnspan=2, padx=(0, 10), pady=(5, 0), sticky="ew")
        
        browse_btn = ctk.CTkButton(
            path_container,
            text="📁 Browse",
            command=self.browse_dataset_path,
            width=120,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#3b8ed0", "#1f6aa5"),
            hover_color=("#2d6da3", "#144870")
        )
        browse_btn.grid(row=1, column=2, pady=(5, 0))
        
        # Status indicator
        self.path_status = ctk.CTkLabel(
            path_container,
            text="⚠️ No dataset selected",
            font=ctk.CTkFont(size=12),
            text_color=("orange", "orange")
        )
        self.path_status.grid(row=2, column=0, columnspan=3, pady=(5, 0))
        
    def browse_dataset_path(self):
        """Open file dialog to select dataset folder"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_path.set(folder)
            self.path_status.configure(
                text="✅ Dataset path configured",
                text_color=("green", "lightgreen")
            )
    
    def create_tabview(self):
        """Create tabview for different modes"""
        self.tabview = ctk.CTkTabview(
            self,
            corner_radius=15,
            border_width=2,
            segmented_button_selected_color=("#3b8ed0", "#1f6aa5"),
            segmented_button_selected_hover_color=("#2d6da3", "#144870")
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        
        # Create tabs
        self.tab_interactive = self.tabview.add("🔍 Interactive Mode")
        self.tab_global = self.tabview.add("📊 Global Analysis")
        
        # Setup tabs
        self.setup_interactive_tab()
        self.setup_global_tab()
    
    def setup_interactive_tab(self):
        """Setup interactive mode tab"""
        self.tab_interactive.grid_rowconfigure(1, weight=1)
        self.tab_interactive.grid_columnconfigure(0, weight=1)
        
        # Image selection frame
        selection_frame = ctk.CTkFrame(self.tab_interactive, fg_color="transparent")
        selection_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        selection_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Reference image card
        ref_card = ctk.CTkFrame(selection_frame, fg_color=("gray85", "gray20"), corner_radius=10)
        ref_card.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        
        ref_label = ctk.CTkLabel(
            ref_card,
            text="📷 Reference Image",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ref_label.pack(pady=(15, 5))
        
        self.ref_status_label = ctk.CTkLabel(
            ref_card,
            text="No image selected",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60")
        )
        self.ref_status_label.pack(pady=(0, 10))
        
        self.select_ref_btn = ctk.CTkButton(
            ref_card,
            text="📁 Select Reference",
            command=self.select_reference_image,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#2fa572", "#106a43"),
            hover_color=("#207d54", "#0d4f31")
        )
        self.select_ref_btn.pack(padx=15, pady=(0, 15), fill="x")
        
        # Degraded image card
        deg_card = ctk.CTkFrame(selection_frame, fg_color=("gray85", "gray20"), corner_radius=10)
        deg_card.grid(row=0, column=1, padx=(10, 0), pady=5, sticky="ew")
        
        deg_label = ctk.CTkLabel(
            deg_card,
            text="🔧 Degraded Image",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        deg_label.pack(pady=(15, 5))
        
        self.deg_status_label = ctk.CTkLabel(
            deg_card,
            text="Select reference first",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60")
        )
        self.deg_status_label.pack(pady=(0, 10))
        
        self.select_deg_btn = ctk.CTkButton(
            deg_card,
            text="📁 Select Degraded",
            command=self.select_degraded_image,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#d0861f", "#8f5d15"),
            hover_color=("#a86b19", "#6b4710"),
            state="disabled"
        )
        self.select_deg_btn.pack(padx=15, pady=(0, 15), fill="x")
        
        # Results display
        results_label = ctk.CTkLabel(
            self.tab_interactive,
            text="📈 Quality Assessment Results",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        results_label.grid(row=1, column=0, sticky="w", padx=25, pady=(10, 5))
        
        self.interactive_results = ctk.CTkTextbox(
            self.tab_interactive,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="none",
            corner_radius=10,
            border_width=2
        )
        self.interactive_results.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.tab_interactive.grid_rowconfigure(2, weight=1)
        
        welcome_text = (
            "Welcome to Interactive Mode! 🎉\n\n"
            "This mode allows you to compare individual images:\n"
            "  • Select any reference image from your dataset\n"
            "  • Choose a degraded version to compare\n"
            "  • Get instant quality metrics (SSIM and MS-SSIM)\n"
            "  • See detailed interpretation of results\n\n"
            "Ready to start? Select your dataset path above and choose a reference image!"
        )
        self.interactive_results.insert("1.0", welcome_text)
    
    def setup_global_tab(self):
        """Setup global mode tab"""
        self.tab_global.grid_rowconfigure(2, weight=1)
        self.tab_global.grid_columnconfigure(0, weight=1)
        
        # Control panel
        control_frame = ctk.CTkFrame(self.tab_global, fg_color=("gray85", "gray20"), corner_radius=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        self.start_global_btn = ctk.CTkButton(
            control_frame,
            text="▶️ Start global Analysis",
            command=self.start_global_analysis,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#2fa572", "#106a43"),
            hover_color=("#207d54", "#0d4f31")
        )
        self.start_global_btn.pack(padx=20, pady=(20, 15))
        
        # Progress section
        progress_container = ctk.CTkFrame(control_frame, fg_color="transparent")
        progress_container.pack(fill="x", padx=20, pady=(0, 20))
        
        self.progress_label = ctk.CTkLabel(
            progress_container,
            text="⏳ Ready to start analysis",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.progress_label.pack(pady=(0, 8))
        
        self.progress_bar = ctk.CTkProgressBar(progress_container, height=20, corner_radius=10)
        self.progress_bar.pack(fill="x", pady=(0, 8))
        self.progress_bar.set(0)
        
        self.progress_percent = ctk.CTkLabel(
            progress_container,
            text="0%",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray60")
        )
        self.progress_percent.pack()
        
        # Results display
        results_label = ctk.CTkLabel(
            self.tab_global,
            text="📋 Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        results_label.grid(row=1, column=0, sticky="w", padx=25, pady=(10, 5))
        
        self.global_results = ctk.CTkTextbox(
            self.tab_global,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="none",
            corner_radius=10,
            border_width=2
        )
        self.global_results.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        welcome_text = (
            "Welcome to global Analysis Mode! \n\n"
            "This mode processes your entire dataset:\n"
            "  • Calculates SSIM for all image pairs\n"
            "  • Calculates MS-SSIM for all image pairs\n"
            "  • Computes performance metrics (PRCC, SROC, RMSE)\n"
            "  • Generates comprehensive comparison report\n\n"
            "Ready to start? Set your dataset path and click 'Start global Analysis'!"
        )
        self.global_results.insert("1.0", welcome_text)
    
    def select_reference_image(self):
        """Select reference image"""
        if not self.dataset_path.get():
            messagebox.showwarning("⚠️ Warning", "Please set the dataset path first!")
            return
        
        ref_path = os.path.join(self.dataset_path.get(), "refimgs")
        if not os.path.exists(ref_path):
            messagebox.showerror("❌ Error", f"Reference images folder not found:\n{ref_path}")
            return
        
        filepath = filedialog.askopenfilename(
            initialdir=ref_path,
            title="Select Reference Image",
            filetypes=[
                ("Image files", "*.bmp *.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.ref_image_path = filepath
            self.ref_image_name = os.path.basename(filepath)
            self.ref_image = cv2.imread(filepath)
            
            if self.ref_image is None:
                messagebox.showerror("❌ Error", "Could not load the reference image!")
                return
            
            # Update UI
            self.select_deg_btn.configure(state="normal")
            self.ref_status_label.configure(
                text=f"✓ {self.ref_image_name}",
                text_color=("green", "lightgreen")
            )
            self.deg_status_label.configure(
                text="Ready to select degraded image",
                text_color=("orange", "yellow")
            )
            
            result_text = (
                f"✅ Reference Image Selected!\n"
                f"{'='*70}\n\n"
                f"📄 Filename: {self.ref_image_name}\n"
                f"📐 Dimensions: {self.ref_image.shape[1]} × {self.ref_image.shape[0]} pixels\n"
                f"🎨 Channels: {self.ref_image.shape[2]}\n\n"
                f"{'='*70}\n\n"
                f"👉 Next step: Click 'Select Degraded' to choose a degraded version.\n"
                f"   Only images related to this reference will be shown."
            )
            
            self.interactive_results.delete("1.0", "end")
            self.interactive_results.insert("1.0", result_text)
    
    def select_degraded_image(self):
        """Select degraded image - showing only relevant ones"""
        if not self.dataset_path.get() or self.ref_image is None:
            return
        
        dataset_path = self.dataset_path.get()
        
        # Get the file paths structure
        try:
            images_dict = file_paths()
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error loading file paths:\n{str(e)}")
            return
        
        # Check if reference image has degraded versions
        if self.ref_image_name not in images_dict:
            messagebox.showwarning(
                "⚠️ Warning",
                f"No degraded versions found for:\n{self.ref_image_name}\n\n"
                "You can still select any image manually."
            )
            
            filepath = filedialog.askopenfilename(
                initialdir=dataset_path,
                title="Select Degraded Image",
                filetypes=[
                    ("Image files", "*.bmp *.jpg *.jpeg *.png"),
                    ("All files", "*.*")
                ]
            )
            
            if filepath:
                self.process_degraded_image(filepath)
            return
        
        # Create filtered view with only relevant images
        degraded_images = []
        for deg_type, deg_images_list in images_dict[self.ref_image_name].items():
            deg_image_names = [deg_images_list[i] for i in range(0, len(deg_images_list), 2)]
            for deg_img_name in deg_image_names:
                degraded_images.append({
                    'name': deg_img_name,
                    'type': deg_type,
                    'path': os.path.join(dataset_path, deg_type, deg_img_name)
                })
        
        if not degraded_images:
            messagebox.showwarning("⚠️ Warning", "No degraded images found!")
            return
        
        # Create temporary directory with filtered images
        self.temp_dir = tempfile.mkdtemp(prefix="iqa_filtered_")
        
        try:
            # Copy only relevant images to temp directory
            for deg_info in degraded_images:
                if os.path.exists(deg_info['path']):
                    dest_folder = os.path.join(self.temp_dir, deg_info['type'])
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_path = os.path.join(dest_folder, deg_info['name'])
                    shutil.copy2(deg_info['path'], dest_path)
            
            # Show info about filtering (optional - can be removed if too intrusive)
            # deg_types = list(set([d['type'] for d in degraded_images]))
            # messagebox.showinfo(
            #     "📂 Filtered View",
            #     f"Showing {len(degraded_images)} degraded versions of:\n"
            #     f"{self.ref_image_name}\n\n"
            #     f"Degradation types available:\n" + 
            #     "\n".join([f"  • {dt.upper()}" for dt in sorted(deg_types)])
            # )
            
            # Open file dialog with filtered images
            filepath = filedialog.askopenfilename(
                initialdir=self.temp_dir,
                title=f"Select Degraded Version of {self.ref_image_name}",
                filetypes=[
                    ("Image files", "*.bmp *.jpg *.jpeg *.png"),
                    ("All files", "*.*")
                ]
            )
            
            if filepath:
                # Find original path and type
                selected_name = os.path.basename(filepath)
                original_path = None
                deg_type = "unknown"
                
                for deg_info in degraded_images:
                    if deg_info['name'] == selected_name:
                        original_path = deg_info['path']
                        deg_type = deg_info['type']
                        break
                
                if original_path:
                    self.process_degraded_image(original_path, deg_type)
        
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error creating filtered view:\n{str(e)}")
        
        finally:
            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                except:
                    pass
                self.temp_dir = None
    
    def process_degraded_image(self, filepath, deg_type=None):
        """Process the selected degraded image"""
        self.deg_image_path = filepath
        deg_name = os.path.basename(filepath)
        self.deg_image = cv2.imread(filepath)
        
        if self.deg_image is None:
            messagebox.showerror("❌ Error", "Could not load the degraded image!")
            return
        
        # Detect degradation type if not provided
        if deg_type is None:
            for dtype in ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']:
                if dtype in filepath.lower():
                    deg_type = dtype
                    break
            if deg_type is None:
                deg_type = "unknown"
        
        self.deg_type = deg_type
        
        # Update UI
        self.deg_status_label.configure(
            text=f"✓ {deg_name}",
            text_color=("green", "lightgreen")
        )
        
        # Resize if needed
        if self.deg_image.shape != self.ref_image.shape:
            self.deg_image = cv2.resize(self.deg_image, (self.ref_image.shape[1], self.ref_image.shape[0]))
        
        # Calculate metrics
        self.calculate_interactive_metrics(deg_name)
    
    def calculate_interactive_metrics(self, deg_name):
        """Calculate and display metrics for interactive mode"""
        self.interactive_results.delete("1.0", "end")
        self.interactive_results.insert("1.0", "⚙️ Calculating quality metrics...\nPlease wait...")
        self.update()
        
        # Calculate metrics for this specific image pair only
        ssim_value = calc_ssim(self.ref_image, self.deg_image)
        msssim_value = calc_msssim(self.ref_image, self.deg_image)
        
        # Format results with better styling
        result_text = "="*80 + "\n"
        result_text += "🎯 QUALITY ASSESSMENT RESULTS\n"
        result_text += "="*80 + "\n\n"
        
        result_text += "📊 IMAGE INFORMATION\n"
        result_text += "-"*80 + "\n"
        result_text += f"  Reference:  {self.ref_image_name}\n"
        result_text += f"  Degraded:   {deg_name}\n"
        result_text += f"  Type:       {self.deg_type.upper()}\n"
        result_text += f"  Size:       {self.ref_image.shape[1]} × {self.ref_image.shape[0]} pixels\n"
        result_text += "\n"
        
        result_text += "="*80 + "\n"
        result_text += "📈 QUALITY METRICS FOR THIS IMAGE PAIR\n"
        result_text += "="*80 + "\n\n"
        result_text += f"  {'Metric':<15} {'Value':<15} {'Quality Range':<30}\n"
        result_text += f"  {'-'*14} {'-'*14} {'-'*29}\n"
        result_text += f"  {'SSIM':<15} {ssim_value:<15.6f} [0.0 - 1.0, higher = better]\n"
        result_text += f"  {'MS-SSIM':<15} {msssim_value:<15.6f} [0.0 - 1.0, higher = better]\n"
        result_text += "\n"
        
        # Interpretation
        result_text += "="*80 + "\n"
        result_text += " QUALITY INTERPRETATION\n"
        result_text += "="*80 + "\n\n"
        
        # def interpret_quality(score):
        #     if score >= 0.95:
        #         return ("Excellent", "⭐⭐⭐⭐⭐", "Almost imperceptible degradation")
        #     elif score >= 0.90:
        #         return ("Very Good", "⭐⭐⭐⭐", "Minor degradation")
        #     elif score >= 0.80:
        #         return ("Good", "⭐⭐⭐", "Noticeable but acceptable")
        #     elif score >= 0.70:
        #         return ("Fair", "⭐⭐", "Moderate degradation")
        #     elif score >= 0.50:
        #         return ("Poor", "⭐", "Significant degradation")
        #     else:
        #         return ("Very Poor", "☆", "Severe degradation")
        
        # ssim_quality, ssim_stars, ssim_desc = interpret_quality(ssim_value)
        # msssim_quality, msssim_stars, msssim_desc = interpret_quality(msssim_value)
        
        result_text += f"SSIM Assessment:\n"
        result_text += f"  Score:       {ssim_value:.4f}\n"
        # result_text += f"  Rating:      {ssim_quality} {ssim_stars}\n"
        # result_text += f"  Description: {ssim_desc}\n\n"
        
        result_text += f"MS-SSIM Assessment:\n"
        result_text += f"  Score:       {msssim_value:.4f}\n"
        # result_text += f"  Rating:      {msssim_quality} {msssim_stars}\n"
        # result_text += f"  Description: {msssim_desc}\n\n"
        
        # Comparison
        result_text += "-"*80 + "\n"
        result_text += "METRIC COMPARISON\n"
        result_text += "-"*80 + "\n"
        
        diff = abs(ssim_value - msssim_value)
        if diff > 0.05:
            if msssim_value > ssim_value:
                result_text += f"📈 MS-SSIM rates this image {(msssim_value - ssim_value)*100:.2f}% higher\n"
                result_text += "   → Multi-scale analysis captures quality aspects better\n"
            else:
                result_text += f"📉 SSIM rates this image {(ssim_value - msssim_value)*100:.2f}% higher\n"
                result_text += "   → Single-scale features are more relevant here\n"
        else:
            result_text += "✓ Both metrics agree closely (difference < 5%)\n"
            result_text += "  → Consistent quality assessment\n"
        
        result_text += "\n" + "="*80 + "\n"
        result_text += "✅ Analysis Complete!\n"
        result_text += "="*80 + "\n\n"
        result_text += " TIP: Use 'global Analysis' mode to see PRCC, SROC, and RMSE performance\n"
        result_text += "   metrics across the entire dataset."
        
        self.interactive_results.delete("1.0", "end")
        self.interactive_results.insert("1.0", result_text)
    
    def start_global_analysis(self):
        """Start global analysis in a separate thread"""
        if not self.dataset_path.get():
            messagebox.showwarning("⚠️ Warning", "Please set the dataset path first!")
            return
        
        # Verify required folders exist
        ref_path = os.path.join(self.dataset_path.get(), "refimgs")
        dmos_path = os.path.join(self.dataset_path.get(), "dmos.mat")
        
        if not os.path.exists(ref_path):
            messagebox.showerror("❌ Error", f"Reference images folder not found:\n{ref_path}")
            return
        
        if not os.path.exists(dmos_path):
            messagebox.showerror("❌ Error", f"DMOS file not found:\n{dmos_path}")
            return
        
        self.start_global_btn.configure(state="disabled", text="⏳ Analysis Running...")
        self.global_results.delete("1.0", "end")
        self.global_results.insert("1.0", " Starting global analysis...\n\nThis may take several minutes.\nPlease wait...")
        self.progress_bar.set(0)
        self.progress_percent.configure(text="0%")
        self.progress_label.configure(text="⏳ Initializing analysis...")
        
        # Run in separate thread
        thread = threading.Thread(target=self.run_global_analysis, daemon=True)
        thread.start()
    
    def run_global_analysis(self):
        """Run the global analysis"""
        try:
            # Update progress
            self.update_progress(0.05, "📊 Calculating SSIM for all images...")
            
            ssim_results = self.calculate_all_metrics_silent(calc_ssim, "SSIM")
            
            self.update_progress(0.35, "🔍 Calculating MS-SSIM for all images...")
            
            msssim_results = self.calculate_all_metrics_silent(calc_msssim, "MS-SSIM")
            
            self.update_progress(0.65, "📋 Extracting DMOS values...")
            
            dmos_arrays = self.extract_dmos_arrays_silent(ssim_results)
            
            self.update_progress(0.75, "📈 Calculating SSIM performance metrics...")
            
            ssim_metrics = self.calculate_performance_metrics_silent(ssim_results, dmos_arrays, "SSIM")
            
            self.update_progress(0.85, "📈 Calculating MS-SSIM performance metrics...")
            
            msssim_metrics = self.calculate_performance_metrics_silent(msssim_results, dmos_arrays, "MS-SSIM")
            
            self.update_progress(0.95, "📝 Generating comprehensive report...")
            
            report = self.generate_global_report(ssim_results, msssim_results, ssim_metrics, msssim_metrics)
            
            self.update_progress(1.0, "✅ Analysis complete!")
            
            # Display results
            self.after(0, lambda: self.global_results.delete("1.0", "end"))
            self.after(0, lambda: self.global_results.insert("1.0", report))
            self.after(0, lambda: self.start_global_btn.configure(state="normal", text="▶️ Start global Analysis"))
            
            # Show completion message
            self.after(0, lambda: messagebox.showinfo(
                "✅ Success",
                "global analysis completed successfully!\n\nReview the results in the text area below."
            ))
            
        except Exception as e:
            error_msg = f"Error during global analysis:\n{str(e)}"
            self.after(0, lambda: messagebox.showerror("❌ Error", error_msg))
            self.after(0, lambda: self.start_global_btn.configure(state="normal", text="▶️ Start global Analysis"))
            self.after(0, lambda: self.progress_label.configure(text="❌ Analysis failed!"))
    
    def update_progress(self, value, text):
        """Update progress bar and label"""
        self.after(0, lambda: self.progress_bar.set(value))
        self.after(0, lambda: self.progress_label.configure(text=text))
        self.after(0, lambda: self.progress_percent.configure(text=f"{int(value*100)}%"))
    
    def calculate_all_metrics_silent(self, metric_function, metric_name):
        """Calculate metrics without printing individual values"""
        images_dict = file_paths()
        
        ref_path = os.path.join(self.dataset_path.get(), "refimgs")
        dataset_path = self.dataset_path.get()
        
        metric_arrays = {
            'fastfading': [],
            'gblur': [],
            'jp2k': [],
            'jpeg': [],
            'wn': []
        }
        
        for ref_image_name, degradations in images_dict.items():
            ref_image_path = os.path.join(ref_path, ref_image_name)
            ref_img = cv2.imread(ref_image_path)
            
            if ref_img is None:
                continue
            
            for deg_type, deg_images_list in degradations.items():
                deg_image_names = [deg_images_list[i] for i in range(0, len(deg_images_list), 2)]
                
                for deg_img_name in deg_image_names:
                    deg_image_path = os.path.join(dataset_path, deg_type, deg_img_name)
                    deg_img = cv2.imread(deg_image_path)
                    
                    if deg_img is None:
                        continue
                    
                    if deg_img.shape != ref_img.shape:
                        deg_img = cv2.resize(deg_img, (ref_img.shape[1], ref_img.shape[0]))
                    
                    metric_value = metric_function(ref_img, deg_img)
                    metric_arrays[deg_type].append(metric_value)
        
        global_metric_array = []
        for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
            global_metric_array.extend(metric_arrays[deg_type])
        
        metric_arrays['global'] = global_metric_array
        return metric_arrays
    
    def extract_dmos_arrays_silent(self, metric_results):
        """Extract DMOS arrays without printing"""
        mat_file_path = os.path.join(self.dataset_path.get(), "dmos.mat")
        mat_data = scipy.io.loadmat(mat_file_path)
        
        dmos_global = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                if isinstance(mat_data[key], np.ndarray):
                    dmos_global = mat_data[key][0]
                    break
        
        if dmos_global is None:
            raise ValueError("Could not find DMOS data in mat file")
        
        dmos_arrays = {
            'jp2k': [],
            'jpeg': [],
            'wn': [],
            'gblur': [],
            'fastfading': [],
            'global': dmos_global.tolist()
        }
        
        counts = {
            'jp2k': len(metric_results['jp2k']),
            'jpeg': len(metric_results['jpeg']),
            'wn': len(metric_results['wn']),
            'gblur': len(metric_results['gblur']),
            'fastfading': len(metric_results['fastfading'])
        }
        
        start_idx = 0
        for deg_type in ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']:
            count = counts[deg_type]
            dmos_arrays[deg_type] = dmos_global[start_idx:start_idx + count].tolist()
            start_idx += count
        
        return dmos_arrays
    
    def calculate_performance_metrics_silent(self, metric_results, dmos_results, metric_name):
        """Calculate performance metrics without printing"""
        metrics = {}
        deg_types = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading', 'global']
        
        for deg_type in deg_types:
            metric_array = metric_results[deg_type]
            dmos_array = dmos_results[deg_type]
            
            if len(metric_array) != len(dmos_array) or len(metric_array) < 2:
                metrics[deg_type] = {'prcc': None, 'sroc': None, 'rmse': None}
                continue
            
            try:
                prcc = calculate_prcc(metric_array, dmos_array)
                sroc = calculate_sroc(metric_array, dmos_array)
                rmse = calculate_rmse(metric_array, dmos_array)
                
                metrics[deg_type] = {
                    'prcc': prcc,
                    'sroc': sroc,
                    'rmse': rmse
                }
            except Exception as e:
                metrics[deg_type] = {'prcc': None, 'sroc': None, 'rmse': None}
        
        return metrics
    
    def generate_global_report(self, ssim_results, msssim_results, ssim_metrics, msssim_metrics):
        """Generate comprehensive global analysis report"""
        report = "="*80 + "\n"
        report += "🎉 global ANALYSIS COMPLETE\n"
        report += "="*80 + "\n\n"
        
        # Dataset info
        report += "📁 DATASET INFORMATION\n"
        report += "-"*80 + "\n"
        report += f"Location: {self.dataset_path.get()}\n"
        report += f"Total images analyzed: {len(ssim_results['global'])}\n"
        report += "\n"
        
        # Summary statistics
        report += "="*80 + "\n"
        report += "📊 SUMMARY STATISTICS\n"
        report += "="*80 + "\n\n"
        
        deg_types = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
        
        for metric_name, results in [("SSIM", ssim_results), ("MS-SSIM", msssim_results)]:
            report += f"{'─'*80}\n"
            report += f"{metric_name} SCORES BY DEGRADATION TYPE\n"
            report += f"{'─'*80}\n\n"
            
            report += f"{'Type':<12} {'Count':<8} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std Dev':<10}\n"
            report += f"{'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9}\n"
            
            for deg_type in deg_types:
                arr = results[deg_type]
                if len(arr) > 0:
                    report += f"{deg_type.upper():<12} {len(arr):<8} {np.min(arr):<10.4f} {np.max(arr):<10.4f} {np.mean(arr):<10.4f} {np.std(arr):<10.4f}\n"
            
            arr = results['global']
            report += f"{'-'*72}\n"
            report += f"{'GLOBAL':<12} {len(arr):<8} {np.min(arr):<10.4f} {np.max(arr):<10.4f} {np.mean(arr):<10.4f} {np.std(arr):<10.4f}\n"
            report += "\n"
        
        # Performance metrics tables
        report += "\n" + "="*80 + "\n"
        report += "📈 PERFORMANCE METRICS\n"
        report += "="*80 + "\n\n"
        
        for metric_name, metrics in [("SSIM", ssim_metrics), ("MS-SSIM", msssim_metrics)]:
            report += f"{'─'*80}\n"
            report += f"{metric_name} PERFORMANCE\n"
            report += f"{'─'*80}\n\n"
            report += f"{'Degradation':<15} {'PRCC':<15} {'SROC':<15} {'RMSE':<15}\n"
            report += f"{'-'*15} {'-'*14} {'-'*14} {'-'*14}\n"
            
            for deg_type in deg_types + ['global']:
                if deg_type in metrics and metrics[deg_type]['prcc'] is not None:
                    prcc = metrics[deg_type]['prcc']
                    sroc = metrics[deg_type]['sroc']
                    rmse = metrics[deg_type]['rmse']
                    report += f"{deg_type.upper():<15} {prcc:<15.4f} {sroc:<15.4f} {rmse:<15.4f}\n"
                else:
                    report += f"{deg_type.upper():<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}\n"
            report += "\n"
        
        # Comparison
        report += "\n" + "="*80 + "\n"
        report += "⚖️  COMPARISON: SSIM vs MS-SSIM\n"
        report += "="*80 + "\n\n"
        
        # PRCC Comparison
        report += "PRCC (Pearson Correlation) - Higher is Better\n"
        report += "-"*80 + "\n"
        report += f"{'Degradation':<15} {'SSIM':<15} {'MS-SSIM':<15} {'Δ':<15} {'Winner':<15}\n"
        report += "-"*80 + "\n"
        
        prcc_improvements = []
        for deg_type in deg_types + ['global']:
            if (deg_type in ssim_metrics and ssim_metrics[deg_type]['prcc'] is not None and
                deg_type in msssim_metrics and msssim_metrics[deg_type]['prcc'] is not None):
                
                ssim_prcc = ssim_metrics[deg_type]['prcc']
                msssim_prcc = msssim_metrics[deg_type]['prcc']
                improvement = msssim_prcc - ssim_prcc
                
                if abs(improvement) < 0.001:
                    winner = "TIE ≈"
                else:
                    winner = "MS-SSIM ↑" if msssim_prcc > ssim_prcc else "SSIM ↑"
                
                prcc_improvements.append(improvement)
                report += f"{deg_type.upper():<15} {ssim_prcc:<15.4f} {msssim_prcc:<15.4f} {improvement:+<15.4f} {winner:<15}\n"
        
        # SROC Comparison
        report += "\n" + "SROC (Spearman Correlation) - Higher is Better\n"
        report += "-"*80 + "\n"
        report += f"{'Degradation':<15} {'SSIM':<15} {'MS-SSIM':<15} {'Δ':<15} {'Winner':<15}\n"
        report += "-"*80 + "\n"
        
        sroc_improvements = []
        for deg_type in deg_types + ['global']:
            if (deg_type in ssim_metrics and ssim_metrics[deg_type]['sroc'] is not None and
                deg_type in msssim_metrics and msssim_metrics[deg_type]['sroc'] is not None):
                
                ssim_sroc = ssim_metrics[deg_type]['sroc']
                msssim_sroc = msssim_metrics[deg_type]['sroc']
                improvement = msssim_sroc - ssim_sroc
                
                if abs(improvement) < 0.001:
                    winner = "TIE ≈"
                else:
                    winner = "MS-SSIM ↑" if msssim_sroc > ssim_sroc else "SSIM ↑"
                
                sroc_improvements.append(improvement)
                report += f"{deg_type.upper():<15} {ssim_sroc:<15.4f} {msssim_sroc:<15.4f} {improvement:+<15.4f} {winner:<15}\n"
        
        # RMSE Comparison
        report += "\n" + "RMSE (Root Mean Square Error) - Lower is Better\n"
        report += "-"*80 + "\n"
        report += f"{'Degradation':<15} {'SSIM':<15} {'MS-SSIM':<15} {'Δ':<15} {'Winner':<15}\n"
        report += "-"*80 + "\n"
        
        rmse_improvements = []
        for deg_type in deg_types + ['global']:
            if (deg_type in ssim_metrics and ssim_metrics[deg_type]['rmse'] is not None and
                deg_type in msssim_metrics and msssim_metrics[deg_type]['rmse'] is not None):
                
                ssim_rmse = ssim_metrics[deg_type]['rmse']
                msssim_rmse = msssim_metrics[deg_type]['rmse']
                improvement = ssim_rmse - msssim_rmse
                
                if abs(improvement) < 0.001:
                    winner = "TIE ≈"
                else:
                    winner = "MS-SSIM ↓" if msssim_rmse < ssim_rmse else "SSIM ↓"
                
                rmse_improvements.append(improvement)
                report += f"{deg_type.upper():<15} {ssim_rmse:<15.4f} {msssim_rmse:<15.4f} {improvement:+<15.4f} {winner:<15}\n"
        
        # Final verdict
        report += "\n\n" + "="*80 + "\n"
        report += "🏆 FINAL VERDICT\n"
        report += "="*80 + "\n\n"
        
        if prcc_improvements and sroc_improvements and rmse_improvements:
            msssim_wins = (sum(1 for x in prcc_improvements if x > 0) + 
                           sum(1 for x in sroc_improvements if x > 0) + 
                           sum(1 for x in rmse_improvements if x > 0))
            total_comparisons = len(prcc_improvements) + len(sroc_improvements) + len(rmse_improvements)
            
            avg_prcc = np.mean(prcc_improvements)
            avg_sroc = np.mean(sroc_improvements)
            avg_rmse = np.mean(rmse_improvements)
            
            report += f"Overall Statistics:\n"
            report += f"  • MS-SSIM wins: {msssim_wins}/{total_comparisons} comparisons ({(msssim_wins/total_comparisons)*100:.1f}%)\n"
            report += f"  • Average PRCC improvement: {avg_prcc:+.4f}\n"
            report += f"  • Average SROC improvement: {avg_sroc:+.4f}\n"
            report += f"  • Average RMSE improvement: {avg_rmse:+.4f} (positive = MS-SSIM better)\n\n"
            
            if msssim_wins > total_comparisons * 0.6:
                report += "🎯 RESULT: MS-SSIM shows SIGNIFICANTLY BETTER performance!\n"
                report += "   → Multi-scale analysis provides superior quality assessment\n"
            elif msssim_wins > total_comparisons * 0.4:
                report += "⚖️  RESULT: Both metrics perform COMPARABLY\n"
                report += "   → Choice depends on specific use case requirements\n"
            else:
                report += "📊 RESULT: SSIM shows BETTER performance overall\n"
                report += "   → Single-scale features are more relevant for this dataset\n"
        
        report += "\n" + "="*80 + "\n"
        report += "✅ Analysis completed successfully!\n"
        report += "="*80
        
        return report
    
    def on_closing(self):
        """Clean up before closing"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
        self.destroy()


if __name__ == "__main__":
    app = ImageQualityAssessmentApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()