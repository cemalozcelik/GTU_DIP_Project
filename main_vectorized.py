import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt

class Config:
    SAMPLE_DIR = "samples/1312"
    RESULTS_DIR = "results-vectorized"
    
    MIN_SIZE_LARGE_WBC = 200
    MAX_SIZE_LARGE_WBC = 2000
    
    THRESHOLD_BLUE_CHANNEL = 80
    
    CROP_AREA_OUT_WIDTH = 80
    CROP_AREA_OUT_HEIGHT = 80
    
    os.makedirs(f"{RESULTS_DIR}/{SAMPLE_DIR}", exist_ok=True)
    

class PipelineMode:
    SCALE_BLUR_BLUE = "scale_blur_blue"
    BLUR_BLUE_SCALE = "blur_blue_scale"

class FocusMetrics:
    @staticmethod
    def laplacian_variance(gray):
        """The higher the variance, the sharper the image.

        Args:
            gray (numpy.ndarray): Grayscale image.

        Returns:
            float: numpy.ndarray: Laplacian Variance focus measure.
        """
        print("→ Calculating Laplacian Variance...")

        gray = gray.astype(np.float64)
        
        laplacian = np.zeros_like(gray, dtype=np.float64)
        
        laplacian[1:-1, 1:-1] = (
            gray[0:-2, 1:-1] +
            gray[2:, 1:-1] +
            gray[1:-1, 0:-2] +
            gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        
        return laplacian.var()


    @staticmethod
    def tennengrad(gray):
        """ Calculates the Tennengrad focus measure.
            Gx and Gy are computed using Sobel operators.

        Args:
            gray (numpy.ndarray): Grayscale image.

        Returns:
            float: numpy.ndarray: Tennengrad focus measure.
        """
        print("→ Calculating Manual Tennengrad...")

        gray = gray.astype(np.float64)

        gx = (
            -1*gray[:-2, :-2] + 1*gray[:-2, 2:] +
            -2*gray[1:-1, :-2] + 2*gray[1:-1, 2:] +
            -1*gray[2:, :-2] + 1*gray[2:, 2:]
        )

        gy = (
            -1*gray[:-2, :-2] - 2*gray[:-2, 1:-1] - 1*gray[:-2, 2:] +
            1*gray[2:, :-2] + 2*gray[2:, 1:-1] + 1*gray[2:, 2:]
        )

        return np.mean(gx**2 + gy**2)

    @staticmethod
    def brenner(gray):
        """Calculate Brenner focus measure.
            shifted is the image shifted by 2 pixels to the right.

        Args:
            gray (numpy.ndarray): Grayscale image.

        Returns:
            _type_: numpy.ndarray: Brenner focus measure.   
        """
        print("→ Calculating Brenner...")
        shifted = np.roll(gray, -2, axis=1)
        diff = (gray - shifted) ** 2
        return np.sum(diff)
    
    
class FocusAnalysis:
    
    def __init__(self, sample_dir=Config.SAMPLE_DIR):
        self.sample_dir = sample_dir
    
    def analyze(self, save_json=True):
        """Analyze Z-stack images in the sample directory.

        Args:
            save_json (bool): If True, save results to JSON file.

        Returns:
            _type_: list: List of dictionaries with focus metrics for each image.
        """
        results = []

        for file in sorted(os.listdir(self.sample_dir)):
            if not file.lower().endswith(".jpg"):
                continue

            path = os.path.join(self.sample_dir, file)
            img = cv2.imread(path)
            gray = self.gray_scale(img)
            print(f"Processed {file}")

            results.append({
                "file": file,
                "laplacian": float(FocusMetrics.laplacian_variance(gray)),
                "tenengrad": float(FocusMetrics.tennengrad(gray)),
                "brenner": float(FocusMetrics.brenner(gray))
            })
        
        if save_json and results:
            self.save_results_to_json(results)
        
        return results
    
    def save_results_to_json(self, results):
        """Save focus analysis results to JSON file.
        
        Args:
            results (list): List of dictionaries with focus metrics.
        """
        import json
        
        lap = self.normalize([r["laplacian"] for r in results])
        ten = self.normalize([r["tenengrad"] for r in results])
        bre = self.normalize([r["brenner"] for r in results])
        
        for i, r in enumerate(results):
            r["laplacian_normalized"] = float(lap[i])
            r["tenengrad_normalized"] = float(ten[i])
            r["brenner_normalized"] = float(bre[i])
            r["hybrid_score"] = float(0.4 * lap[i] + 0.4 * ten[i] + 0.2 * bre[i])
        
        best_idx = max(range(len(results)), key=lambda i: results[i]["hybrid_score"])
        
        output = {
            "sample_directory": self.sample_dir,
            "total_images": len(results),
            "best_focused_image": {
                "file": results[best_idx]["file"],
                "hybrid_score": results[best_idx]["hybrid_score"]
            },
            "images": results
        }
        
        json_path = f'{Config.RESULTS_DIR}/{Config.SAMPLE_DIR}/focus_analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Focus analysis results saved to {json_path}")
    
    def gray_scale(self, image=None):
        """
        Manual grayscale conversion.
        BGR order is assumed (OpenCV format).
        """
        print("→ Converting to grayscale...")

        if image is None:
            image = self.image

        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        gray = 0.114 * b + 0.587 * g + 0.299 * r

        return gray.astype(np.uint8)


    def normalize(self, values):
        """ Normalize a list of values to the range [0, 1].

        Args:
            values (list): List of numerical values to normalize.

        Returns:
            list: Normalized values in the range [0, 1].
        """
        min_val = min(values)
        max_val = max(values)
        return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0 for v in values]
    
    def print_best_focused(self, results):
        """ Determine and print the best focused image based on hybrid scoring.

        Args:
            results (list): List of dictionaries with focus metrics for each image.

        Returns:
            tuple: Best focused image file name and its hybrid focus score.
        """
        lap = self.normalize([r["laplacian"] for r in results])
        ten = self.normalize([r["tenengrad"] for r in results])
        bre = self.normalize([r["brenner"] for r in results])
        best_score = -1
        best_file = None

        for i, r in enumerate(results):
            score = 0.4 * lap[i] + 0.4 * ten[i] + 0.2 * bre[i]

            if score > best_score:
                best_score = score
                best_file = r["file"]

        return best_file, best_score
    
    def plot_hybrid_focus_curve(self, results):
        """"
        Plot the hybrid focus score curve.
        
        Args:
            results (list): List of dictionaries with focus metrics for each image.
        
        """
        lap = self.normalize([r["laplacian"] for r in results])
        ten = self.normalize([r["tenengrad"] for r in results])
        bre = self.normalize([r["brenner"] for r in results])

        
        files = [r["file"] for r in results]
        hybrid_scores = [0.4 * lap[i] + 0.4 * ten[i] + 0.2 * bre[i] for i in range(len(results))]
        plt.figure(figsize=(12, 6))
        plt.plot(files, hybrid_scores, label='Hybrid Focus Score', marker='o', color='purple')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Image Files')
        plt.ylabel('Hybrid Focus Score')
        plt.title('Hybrid Focus Score Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{Config.RESULTS_DIR}/{Config.SAMPLE_DIR}/hybrid_focus_curve.png')
        plt.close()
        
    
    def plot_focus_curves(self, results):
        """ Plot focus measure curves for Laplacian, Tennengrad, and Brenner.

        Args:
            results (list): List of dictionaries with focus metrics for each image.
        """
        files = [r["file"] for r in results]
        lap = self.normalize([r["laplacian"] for r in results])
        ten = self.normalize([r["tenengrad"] for r in results])
        bre = self.normalize([r["brenner"] for r in results])

        plt.figure(figsize=(12, 6))
        plt.plot(files, lap, label='Laplacian Variance', marker='o')
        plt.plot(files, ten, label='Tennengrad', marker='o')
        plt.plot(files, bre, label='Brenner', marker='o')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Image Files')
        plt.ylabel('Focus Measure Value')
        plt.title('Focus Measure Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{Config.RESULTS_DIR}/{Config.SAMPLE_DIR}/focus_measure_curves.png')
        plt.close()


class WBCellAnalysis:
    
    def __init__(self, image_path, scale=1.0):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.save_image(f"{Config.RESULTS_DIR}/{Config.SAMPLE_DIR}/original_image.png", self.image)
        self.original_image = self.image.copy()
        self.scale = scale
        
        if scale != 1.0:
            self.scaled_image = self.scale_image(self.image, scale)
        else:
            self.scaled_image = self.image.copy()
            
        self.results_file = open(f"{Config.RESULTS_DIR}/{Config.SAMPLE_DIR}/cell_detection_results.txt", "w")

        self.results_file.write(f"Analyzing image: {os.path.basename(image_path)}\n")
    
    def scale_image(self, image, scale):
        """
        Scale the image by the given factor.
        Args:
            image: Input image as a numpy array.
            scale: Scaling factor (e.g., 0.5 for half size).
        Returns:
            Scaled image as a numpy array.
        """
        print("→ Scaling image...")
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        print(f"   New dimensions: {new_width}x{new_height}")
        print(f"   Scaling factor: {scale}")
        
        y_indices = np.minimum((np.arange(new_height) / scale).astype(int), image.shape[0] - 1)
        x_indices = np.minimum((np.arange(new_width) / scale).astype(int), image.shape[1] - 1)
        
        if image.ndim == 2:
            scaled_image = image[y_indices[:, None], x_indices[None, :]]
        else:
            scaled_image = image[y_indices[:, None], x_indices[None, :], :]
        
        return scaled_image
    
    def convolution2d(self, image, kernel):
        """
        Apply 2D convolution to the image using the given kernel.

        Args:
            image: Input image as a numpy array.
            kernel: Convolution kernel as a numpy array.

        Returns:
            Convolved image as a numpy array.
        """
        print("→ Applying 2D Convolution...")
        kernel_height, kernel_width = kernel.shape
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2
        
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        convolved = np.zeros_like(image, dtype=np.float32)
        height = image.shape[0]
        width = image.shape[1]
        
        kernel_flipped = np.flip(kernel)
        
        for c in range(3):
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    window = padded_image[ky:ky+height, kx:kx+width, c]
                    convolved[:, :, c] += window * kernel_flipped[ky, kx]
        
        return convolved.astype(image.dtype)
    
    def blur_image(self, image=None):
        """
        Apply Gaussian blur to the image using separable convolution.

        Args:
            image (_type_, optional): Input image as a numpy array. Defaults to None.

        Returns:
            np.ndarray: Blurred image as a numpy array.
        """
        print("→ Blurring image...")
        if image is None:
            image = self.image
        
        kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        kernel_1d /= kernel_1d.sum()
        
        # Step 1: Horizontal convolution
        pad_w = len(kernel_1d) // 2
        padded = np.pad(image, ((0, 0), (pad_w, pad_w), (0, 0)), mode='edge')
        horizontal = np.zeros_like(image, dtype=np.float32)
        
        for c in range(3):
            for k_idx, k_val in enumerate(kernel_1d):
                horizontal[:, :, c] += padded[:, k_idx:k_idx+image.shape[1], c] * k_val
        
        # Step 2: Vertical convolution
        pad_h = len(kernel_1d) // 2
        padded = np.pad(horizontal, ((pad_h, pad_h), (0, 0), (0, 0)), mode='edge')
        blurred = np.zeros_like(image, dtype=np.float32)
        
        for c in range(3):
            for k_idx, k_val in enumerate(kernel_1d):
                blurred[:, :, c] += padded[k_idx:k_idx+image.shape[0], :, c] * k_val
        
        return blurred.astype(image.dtype)
        
    
    def extract_blue_dominance(self, image, w_r=1.0, w_g=0.0):
        """
        Compute blue dominance image:
        B - w_r * R - w_g * G
        """
        
        print("→ Extracting blue channel dominance...")
        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)

        dominance = blue - w_r * red - w_g * green
        dominance = np.clip(dominance, 0, 255).astype(np.uint8)
        return dominance


    def otsu_threshold(self, img):
        """
        Apply Otsu's thresholding method to create a binary image.
        Args:
            img (np.ndarray): 2D numpy array (grayscale image).
        Returns:
            np.ndarray: Binary image as a 2D numpy array.
        """
        print("→ Applying Otsu's thresholding...")
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        total = img.size
        
        current_max, threshold = 0, 0
        sum_total, sum_foreground = 0, 0
        weight_background, weight_foreground = 0, 0
        
        for i in range(256):
            sum_total += i * hist[i]
        
        for i in range(256):
            weight_background += hist[i]
            if weight_background == 0:
                continue
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break
            
            sum_foreground += i * hist[i]
            
            mean_background = sum_foreground / weight_background
            mean_foreground = (sum_total - sum_foreground) / weight_foreground
            
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            
            if between_class_variance > current_max:
                current_max = between_class_variance
                threshold = i
        print(f"   Otsu's optimal threshold: {threshold}")
        binary = (img >= threshold).astype(np.uint8) * 255
        return binary
    
    def threshold_blue_channel(self, img, threshold=Config.THRESHOLD_BLUE_CHANNEL):
        """
        Threshold the blue channel to create a binary image.
        Args:
            img (np.ndarray): 2D numpy array (blue channel dominance image).
            threshold (int): Threshold value.
        Returns:
            np.ndarray: Binary image as a 2D numpy array.
        """
        print("→ Thresholding blue channel...")
        return (img > threshold).astype(np.uint8) * 255 
    
    def show_image(self, title, image):
        """
        Display the image using matplotlib.
        Args:
            title (str): Title of the image window.
            image (np.ndarray): Image to display as a numpy array.
        """
        plt.figure(figsize=(8, 8))
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def erosion(self, binary, kernel_size=3):
        """Apply Erosion

        Args:
            binary (np.ndarray): 2D numpy array (binary image).
            kernel_size (int, optional): Size of the square structuring element. Defaults to 3.

        Returns:
            np.ndarray: Eroded binary image as a 2D numpy array.
        """
        print("→ Applying erosion...")
        h, w = binary.shape
        result = np.zeros((h, w), dtype=np.uint8)
        offset = kernel_size // 2
        
        padded = np.pad(binary, offset, mode='constant', constant_values=255)
        
        min_val = np.full((h, w), 255, dtype=np.uint8)
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                window = padded[ky:ky+h, kx:kx+w]
                min_val = np.minimum(min_val, window)
        
        return min_val

    def dilation(self, binary, kernel_size=3):
        """Apply Dilation

        Args:
            binary (np.ndarray): 2D numpy array (binary image).
            kernel_size (int, optional): Size of the square structuring element. Defaults to 3.

        Returns:
            np.ndarray: Dilated binary image as a 2D numpy array.
        """
        print("→ Applying dilation...")
        h, w = binary.shape
        result = np.zeros((h, w), dtype=np.uint8)
        offset = kernel_size // 2
        
        padded = np.pad(binary, offset, mode='constant', constant_values=0)
        
        max_val = np.zeros((h, w), dtype=np.uint8)
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                window = padded[ky:ky+h, kx:kx+w]
                max_val = np.maximum(max_val, window)
        
        return max_val

    def opening(self, binary, kernel_size=3):
        """ Apply Opening = erosion + dilation

        Args:
            binary (np.ndarray): 2D numpy array (binary image).
            kernel_size (int, optional): Size of the square structuring element. Defaults to 3.
        Returns:
            np.ndarray: Opened binary image as a 2D numpy array.
        """
        print("→ Applying opening...")
        eroded = self.erosion(binary, kernel_size)
        opened = self.dilation(eroded, kernel_size)
        return opened
    
    def closing(self, binary, kernel_size=3):
        """ Apply Closing = dilation + erosion

        Args:
            binary (np.ndarray): 2D numpy array (binary image).
            kernel_size (int, optional): Size of the square structuring element. Defaults to 3.
        Returns:
            np.ndarray: Closed binary image as a 2D numpy array.
        """     
        print("→ Applying closing...")
        dilated = self.dilation(binary, kernel_size)
        closed = self.erosion(dilated, kernel_size)
        return closed
    
    def connected_components(self, binary):
        """ Apply Connected Components Labeling for binary image.

        Args:
            binary (np.ndarray): 2D numpy array (binary image).

        Returns:
            tuple: Number of labels and labeled image as a 2D numpy array.
        """             
        print("→ Finding connected components...")
        h, w = binary.shape
        labels = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        
        def dfs(y, x, label):
            stack = [(y, x)]
            while len(stack) > 0:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if labels[cy, cx] != 0:
                    continue
                if binary[cy, cx] == 0:
                    continue
                
                labels[cy, cx] = label
                
                stack.append((cy-1, cx))
                stack.append((cy+1, cx))
                stack.append((cy, cx-1))
                stack.append((cy, cx+1))
        
        for y in range(h):
            for x in range(w):
                if binary[y, x] == 255 and labels[y, x] == 0:
                    current_label += 1
                    dfs(y, x, current_label)
        
        return current_label, labels
    
    def find_min_max_area(self, labels, num_labels):
        """ Find minimum and maximum area of connected components.

        Args:
            labels (np.ndarray): 2D numpy array with labeled connected components.
            num_labels (int): Number of labels found.
        Returns:
            tuple: Minimum and maximum area of connected components.
        """
        areas = []
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            areas.append(area)
        
        min_area = min(areas) if areas else 0
        max_area = max(areas) if areas else 0
        print(f"   Min area: {min_area}, Max area: {max_area}")
        return min_area, max_area
    
    def filter_by_area(self, labels, num_labels, min_area, max_area):
        """ Filter connected components by area.

        Args:
            labels (np.ndarray): 2D numpy array with labeled connected components.
            num_labels (int): Number of labels found.
            min_area (int): Minimum area of components to keep.
            max_area (int): Maximum area of components to keep.

        Returns:
            np.ndarray: Filtered binary image as a 2D numpy array.
        """                         
        print(f"→ Filtering by area ({min_area}-{max_area})...")
        filtered = np.zeros_like(labels, dtype=np.uint8)
        
        valid_count = 0
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if min_area <= area <= max_area:
                filtered[labels == label] = 255
                valid_count += 1
        
        print(f"   {valid_count} valid cells found")
        return filtered
    
    def crop_center_window(self, image, center_y, center_x,
                        out_h=Config.CROP_AREA_OUT_HEIGHT,
                        out_w=Config.CROP_AREA_OUT_WIDTH):
        """
        Crop a fixed-size window from the image centered at (center_y, center_x).
        If the window goes outside bounds, missing parts are filled with black padding.

        Args:
            image (np.ndarray): Original image (H,W,3)
            center_y (int): Center y (row)
            center_x (int): Center x (col)
            out_h (int): Output height
            out_w (int): Output width

        Returns:
            np.ndarray: Cropped window of shape (out_h, out_w, 3)
        """
        H, W = image.shape[:2]

        half_h = out_h // 2
        half_w = out_w // 2

        y1 = center_y - half_h
        y2 = center_y + half_h
        x1 = center_x - half_w
        x2 = center_x + half_w

        out = np.zeros((out_h, out_w, 3), dtype=image.dtype)

        src_y1 = max(0, y1)
        src_y2 = min(H, y2)
        src_x1 = max(0, x1)
        src_x2 = min(W, x2)

        dst_y1 = src_y1 - y1
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = src_x1 - x1
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        out[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        return out
    
    def crop_cells_by_area_center_window(self, image, labels, num_labels, min_area, max_area, scale=1.0):
        """Crop cells by area using center window approach.
        
        Args:
            image (np.ndarray): Original full-size image to crop from
            labels (np.ndarray): Label image (may be scaled)
            num_labels (int): Number of labels
            min_area (int): Minimum area
            max_area (int): Maximum area
            scale (float): Scale factor used for labels
        """
        crops = []
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if min_area <= area <= max_area:
                ys, xs = np.where(labels == label)
                
                if len(ys) == 0 or len(xs) == 0:
                    continue

                cy_scaled = int(np.mean(ys))
                cx_scaled = int(np.mean(xs))
                
                cy_original = int(cy_scaled / scale)
                cx_original = int(cx_scaled / scale)

                patch = self.crop_center_window(image, cy_original, cx_original)
                crops.append(patch)

        return crops
    
    def overlay_labels_on_image(self, image, labels):
        """ Overlay connected component labels on the original image.

        Args:
            image (np.ndarray): Original image as a numpy array.
            labels (np.ndarray): 2D numpy array with labeled connected components.

        Returns:
            np.ndarray: Image with labels overlaid.
        """             
        print("→ Overlaying labels on image...")
        overlay = image.copy()
        
        mask = labels > 0
        overlay[mask] = [0, 0, 255]

        return overlay

    def analyze_image(
        self,
        scale=1.0,
        pipeline_mode=PipelineMode.SCALE_BLUR_BLUE,
        w_r=1.0,
        w_g=0.0,
        show_figures=False
    ):
        """Main analysis pipeline with organized folder structure."""
        self.image = self.original_image.copy()

        if pipeline_mode == PipelineMode.SCALE_BLUR_BLUE:
            # Pipeline 1: SCALE → BLUR → BLUE
            if scale != 1.0:
                self.image = self.scale_image(self.image, scale)
            self.blurred = self.blur_image(self.image)
            blue_dom = self.extract_blue_dominance(
                self.blurred, w_r=w_r, w_g=w_g
            )
        else:
            # Pipeline 2: BLUR → BLUE → SCALE
            self.blurred = self.blur_image(self.image)
            blue_dom = self.extract_blue_dominance(
                self.blurred, w_r=w_r, w_g=w_g
            )
            if scale != 1.0:
                blue_dom = self.scale_image(blue_dom, scale)

        weights_folder = f"wr_{w_r}_wg_{w_g}"
        result_base_path = f"{Config.RESULTS_DIR}/{Config.SAMPLE_DIR}/{pipeline_mode}/{weights_folder}"
        os.makedirs(result_base_path, exist_ok=True)
        
        tag = f"wr_{w_r}_wg_{w_g}_{pipeline_mode}"
        
        self.save_image(
            f"{result_base_path}/blue_dom.png",
            blue_dom
        )          
        blue_channel = blue_dom
        
        otsu_thresholded = self.otsu_threshold(blue_channel)
        self.save_image(f"{result_base_path}/otsu_thresholded_blue_channel.png", otsu_thresholded)
        
        binary = otsu_thresholded
        
        opened = self.opening(binary, kernel_size=3)
        self.save_image(f"{result_base_path}/after_opening.png", opened)
            
        closed = self.closing(opened, kernel_size=3)
        self.save_image(f"{result_base_path}/after_closing.png", closed)
        
        num_labels, labels = self.connected_components(closed)
        print(f"   {num_labels} connected components found")
        
        if scale != 1.0:
            overlay_image = self.scale_image(self.original_image, scale)
        else:
            overlay_image = self.original_image.copy()
            
        overlayed_labels = self.overlay_labels_on_image(overlay_image, labels)
        self.save_image(f"{result_base_path}/overlayed_labels.png", overlayed_labels)
        
        large_cells = self.filter_by_area(labels, num_labels, Config.MIN_SIZE_LARGE_WBC, Config.MAX_SIZE_LARGE_WBC)
        self.save_image(f"{result_base_path}/large_white_blood_cells.png", large_cells)
        
        num_large = 0
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if Config.MIN_SIZE_LARGE_WBC <= area <= Config.MAX_SIZE_LARGE_WBC:
                num_large += 1
        
        self.results_file.write(f"[{tag}] Number of large white blood cells detected: {num_large}\n")
        
        min_large, max_large = self.find_min_max_area(labels, num_labels)
        print(f"   Large cells area range: {min_large} - {max_large}")
        
        small_cells = self.filter_by_area(labels, num_labels, 0, Config.MIN_SIZE_LARGE_WBC - 1)
        self.save_image(f"{result_base_path}/small_white_blood_cells.png", small_cells)
        
        num_small = 0
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if area < Config.MIN_SIZE_LARGE_WBC:
                num_small += 1
        
        self.results_file.write(f"[{tag}] Number of small white blood cells detected: {num_small}\n")
        
        large_cells_cropped = self.crop_cells_by_area_center_window(
            self.original_image, labels, num_labels,
            Config.MIN_SIZE_LARGE_WBC, Config.MAX_SIZE_LARGE_WBC, scale
        )
        
        small_cells_cropped = self.crop_cells_by_area_center_window(
            self.original_image, labels, num_labels,
            0, Config.MIN_SIZE_LARGE_WBC - 1, scale
        )
        print(f"   Cropped {len(large_cells_cropped)} large white blood cells")
        print(f"   Cropped {len(small_cells_cropped)} small white blood cells")
        
        mkdir_path_large_cells = f"{result_base_path}/cropped_large_cells"
        os.makedirs(mkdir_path_large_cells, exist_ok=True)
        for idx, cell in enumerate(large_cells_cropped):
            path = os.path.join(mkdir_path_large_cells, f"large_cell_{idx+1}.png")
            self.save_image(path, cell)
            
        mkdir_path_small_cells = f"{result_base_path}/cropped_small_cells"
        os.makedirs(mkdir_path_small_cells, exist_ok=True)
        for idx, cell in enumerate(small_cells_cropped):
            path = os.path.join(mkdir_path_small_cells, f"small_cell_{idx+1}.png")
            self.save_image(path, cell)

        if show_figures:  
            self.show_image("Blurred Image", self.blurred)
            self.show_image("Blue Channel Dominance", blue_channel)
            self.show_image("Thresholded Blue Channel", binary)
            self.show_image("After Opening", opened)
            self.show_image("After Closing", closed)
            self.show_image("Large White Blood Cells", large_cells)
            self.show_image("Small White Blood Cells", small_cells)
        
    def save_image(self, path, image):
        """ Save the image to the specified path.

        Args:
            path (str): Path to save the image.    
            image (np.ndarray): Image to be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        
    def __del__(self):
        """Cleanup: close results file."""
        if hasattr(self, 'results_file') and not self.results_file.closed:
            self.results_file.close()


if __name__ == "__main__":
    
    # Focus Analysis on Z-stack images
    FocusAnalysisInstance = FocusAnalysis()
    results = FocusAnalysisInstance.analyze()
    best_focused_file, best_focus_score = FocusAnalysisInstance.print_best_focused(results)
    FocusAnalysisInstance.plot_focus_curves(results)
    FocusAnalysisInstance.plot_hybrid_focus_curve(results)
    print("\nBest focused image by hybrid scoring:")
    print("File :", best_focused_file)
    print("Score:", best_focus_score)
    image_path = os.path.join(Config.SAMPLE_DIR, best_focused_file)
    
    # White Blood Cell Analysis
    wbca = WBCellAnalysis(image_path, scale=0.5)
    weight_sets = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
    ]

    pipelines = [
        PipelineMode.SCALE_BLUR_BLUE,
        PipelineMode.BLUR_BLUE_SCALE
    ]

    for w_r, w_g in weight_sets:
        for mode in pipelines:
            wbca.analyze_image(
                scale=0.5,
                pipeline_mode=mode,
                w_r=w_r,
                w_g=w_g
            )
    
    print("\nAnalysis complete!")