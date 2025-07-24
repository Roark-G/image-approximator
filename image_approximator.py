import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# PATCH ARRAY
# [patch_id, y ([0-1) : 0 is top 1 is bottom), x ([0-1) : 0 is left 1 is right),
#  angle ([0-1) : 0 is no change 1 is entirely flipped around), 
#  size ([0-1] : 0.5 would be 50% of the image height),
#  recolor_r, recolor_g, recolor_b,
#  recolor_intensity ((0-1] : 0 is no change, 1 is full recolor)]

class ImageApproximatorParameters():
    def __init__(self):
        self.final_patch_pop_size = 200
        
        self.find_patch_pop_size = 100
        self.survival_proportion = 0.25
        
        self.early_stop_eps = 1e-3
        self.early_stop_patience = 15
        self.min_generations = 10
        self.max_patch_generations = 100

        self.mutation_chance = 0.02
        self.mutation_std = 0.05

        self.patch_init_pixel_size = 1

        # options: "white", "average", or a filepath string (e.g., "start.png")
        self.start_mode = "average"

        self.allow_angle = True
        self.allow_resize = True
        self.allow_recolor = True
        self.max_recolor_intensity = 1.0

        self.make_animation = False

class ImageApproximator():
    def __init__(self, target_img_path, patches_dir, parameters = ImageApproximatorParameters()):
        self.parameters = parameters

        self.target_img = self.get_image(target_img_path)
        self.target_img_float = self.target_img.astype(np.float32)
        self.target_img_height = self.target_img.shape[0]
        self.target_img_width = self.target_img.shape[1]

        self.patch_list = self.load_patches(patches_dir)
        self.num_patches = len(self.patch_list)

        if self.parameters.make_animation:
            self.video = []

    def get_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image at path: {path}")
        return img
    
    def get_patches(self, patches_dir):
        patches = []
        for filename in os.listdir(patches_dir):
            if filename.lower().endswith(".png"):
                filepath = os.path.join(patches_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    patches.append(img)
                else:
                    print(f"Warning: Failed to load {filepath}")
        return patches

    def load_patches(self, patches_dir):
        patches = self.get_patches(patches_dir)

        if self.parameters.allow_resize:
            patch_height = self.target_img_height
            return [cv2.resize(patch, (int(patch_height * patch.shape[1] / float(patch.shape[0])), patch_height)) for patch in patches]

        return patches

    def init_working_img(self):
        mode = self.parameters.start_mode

        if mode == "white":
            return np.full((self.target_img_height, self.target_img_width, 3), 255, dtype=np.uint8)
        
        elif mode == "average":
            avg_color = self.target_img.mean(axis=(0, 1))
            return np.full((self.target_img_height, self.target_img_width, 3), avg_color, dtype=np.uint8)

        elif isinstance(mode, str) and os.path.isfile(mode):
            img = self.get_image(mode)
            img_resized = cv2.resize(img, (self.target_img_width, self.target_img_height))
            return img_resized

        else:
            raise ValueError(f"Invalid start_mode: {mode}. Choose 'white', 'average', or a valid image path.")

    def recolor(self, image_bgra, target_color_rgb, intensity):
        intensity = np.clip(intensity, 0.0, 1.0)

        if intensity == 0:
            return image_bgra.copy()

        b_channel, g_channel, r_channel, alpha_channel = cv2.split(image_bgra)

        target_color_bgr = np.array([target_color_rgb[2], target_color_rgb[1], target_color_rgb[0]], dtype=np.uint8)

        colored_bgr = np.zeros_like(image_bgra[:, :, :3])

        colored_bgr[:, :, :] = target_color_bgr

        blended_b = np.where(alpha_channel > 0,
                            (1 - intensity) * b_channel + intensity * target_color_bgr[0],
                            b_channel).astype(np.uint8)
        blended_g = np.where(alpha_channel > 0,
                            (1 - intensity) * g_channel + intensity * target_color_bgr[1],
                            g_channel).astype(np.uint8)
        blended_r = np.where(alpha_channel > 0,
                            (1 - intensity) * r_channel + intensity * target_color_bgr[2],
                            r_channel).astype(np.uint8)

        recolored_image = cv2.merge([blended_b, blended_g, blended_r, alpha_channel])

        return recolored_image

    def make_random_patch(self, completion = 1, error_heatmap = None):
        patch = []
        patch.append(np.random.randint(0, self.num_patches))  # patch index
        # sample (y, x) location
        if error_heatmap is not None:
            # normalize heatmap to a 1D probability distribution
            flat_heatmap = error_heatmap.flatten()
            prob_map = flat_heatmap / flat_heatmap.sum()

            # sample a pixel index according to the heatmap probabilities
            idx = np.random.choice(len(prob_map), p=prob_map)
            h, w = error_heatmap.shape
            y_idx, x_idx = divmod(idx, w)

            # Convert to [0, 1) float coordinates
            patch.append((y_idx + 0.5) / h)  # y
            patch.append((x_idx + 0.5) / w)  # x

        else:
            patch.append(np.random.uniform())  # y
            patch.append(np.random.uniform())  # x

        if self.parameters.allow_angle:
            patch.append(np.random.uniform())  # angle
        if self.parameters.allow_resize:
            total_patches = self.parameters.final_patch_pop_size
            decay_factor = completion if total_patches > 0 else 1.0
            min_size = 0.1 * (1 - decay_factor)  # starts at 0.1, decays to 0
            size = np.random.uniform(min_size, 1.0)
            patch.append(size)
        if self.parameters.allow_recolor:
            patch.extend(np.random.uniform(size=3))  # r, g, b
            patch.append(1 - np.random.uniform())    # intensity

        return np.array(patch)

    def pixelate(self, img, pixel_size):
        h, w = img.shape[:2]
        temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        return pixelated
    
    def make_error_heatmap(self, img, pixelate=True):
        abs_error = abs(img.astype(np.float32) - self.target_img_float)

        if abs_error.ndim == 3:
            abs_error = np.linalg.norm(abs_error, axis=2)  # shape: (H, W)
        
        abs_error += 1e-6
        abs_error /= abs_error.sum()

        if pixelate:
            return self.pixelate(abs_error, self.parameters.patch_init_pixel_size)
        
        return abs_error

    def initialize_patch_pop(self, img, completion = 1):
        error_heatmap = self.make_error_heatmap(img)

        return np.array([
            self.make_random_patch(completion, error_heatmap=error_heatmap)
            for _ in range(self.parameters.find_patch_pop_size)
        ])
    
    def crossover(self, parent1, parent2):
        child = parent1.copy()

        if np.random.rand() < 0.5:
            child[0] = parent2[0]

        for i in range(1, len(child)):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
            else:
                child[i] = (parent1[i] + parent2[i]) / 2
        return child
        
    def mutate_gene_array(self, gene_array):
        """
        Applies Gaussian noise mutation to a gene array, with reflection to keep values in [0, 1].
        The patch ID (index 0) is mutated separately as an integer in the allowed range.
        """
        mutated = gene_array.copy()

        # Mutate patch ID with given probability
        if np.random.rand() < self.parameters.mutation_chance:
            mutated[0] = np.random.randint(0, self.num_patches)

        # Mutate the rest of the genes with given probability
        for i in range(1, len(mutated)):
            if np.random.rand() < self.parameters.mutation_chance:
                mutated[i] += np.random.normal(0, self.parameters.mutation_std)
                # Reflect to stay within [0, 1]
                if mutated[i] < 0:
                    mutated[i] = -mutated[i]
                if mutated[i] > 1:
                    mutated[i] = 2 - mutated[i]
                mutated[i] = max(0, min(1, mutated[i]))  # Clamp just in case

        return mutated
    
    def render_patch(self, base_img, patch_array):
        """
        Overlay the patch on base_img according to parameters in patch_array.

        base_img: np.array (H, W, 3), BGR, uint8
        patch_array: np.array, [patch_id, y, x, angle, size, recolor_r, recolor_g, recolor_b, recolor_intensity]

        Returns a new np.array image with the patch overlaid.
        """
        base_img = base_img.copy()

        patch_id = int(patch_array[0])
        patch_img = self.patch_list[patch_id].copy()  # BGRA

        y_norm = patch_array[1]
        x_norm = patch_array[2]

        angle = patch_array[3] if self.parameters.allow_angle and len(patch_array) > 3 else 0
        size_norm = patch_array[4] if self.parameters.allow_resize and len(patch_array) > 4 else 1.0

        recolor_r = patch_array[5] if self.parameters.allow_recolor and len(patch_array) > 5 else 0
        recolor_g = patch_array[6] if self.parameters.allow_recolor and len(patch_array) > 6 else 0
        recolor_b = patch_array[7] if self.parameters.allow_recolor and len(patch_array) > 7 else 0
        recolor_intensity = patch_array[8] if self.parameters.allow_recolor and len(patch_array) > 8 else 0

        target_height = max(1, int(size_norm * self.target_img_height))  # prevent zero height

        h, w = patch_img.shape[:2]
        if h == 0 or w == 0:
            return base_img  # or handle error gracefully

        aspect_ratio = w / h
        new_w = max(1, int(target_height * aspect_ratio))  # prevent zero width
        patch_img = cv2.resize(patch_img, (new_w, target_height), interpolation=cv2.INTER_AREA)

        center = (patch_img.shape[1] // 2, patch_img.shape[0] // 2)
        rot_angle = angle * 360  # scale normalized angle to degrees
        rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        patch_img = cv2.warpAffine(patch_img, rot_mat, (patch_img.shape[1], patch_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        patch_img = self.recolor(patch_img, (int(recolor_r * 255), int(recolor_g * 255), int(recolor_b * 255)), recolor_intensity)

        base_h, base_w = base_img.shape[:2]
        x_pos = int(x_norm * base_w) - patch_img.shape[1] // 2
        y_pos = int(y_norm * base_h) - patch_img.shape[0] // 2

        base_h, base_w = base_img.shape[:2]
        patch_h, patch_w = patch_img.shape[:2]

        x_pos = int(x_norm * base_w) - patch_w // 2
        y_pos = int(y_norm * base_h) - patch_h // 2

        x_start = max(x_pos, 0)
        y_start = max(y_pos, 0)
        x_end = min(x_pos + patch_w, base_w)
        y_end = min(y_pos + patch_h, base_h)

        patch_x_start = x_start - x_pos
        patch_y_start = y_start - y_pos
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_end = patch_y_start + (y_end - y_start)

        patch_b, patch_g, patch_r, patch_a = cv2.split(patch_img)
        alpha = patch_a[patch_y_start:patch_y_end, patch_x_start:patch_x_end] / 255.0

        base_roi = base_img[y_start:y_end, x_start:x_end]

        for c, patch_channel in enumerate([patch_b, patch_g, patch_r]):
            patch_slice = patch_channel[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            base_roi[:, :, c] = (alpha * patch_slice + (1 - alpha) * base_roi[:, :, c]).astype(np.uint8)

        return base_img
    
    def calculate_mse(self, img1, img2):
        return np.mean((img1 - img2) ** 2)

    def evaluate_patch(self, base_img, patch_array):
        rendered = self.render_patch(base_img, patch_array)
        mse = self.calculate_mse(rendered.astype(np.float32), self.target_img_float)
        return mse
    
    def get_offspring(self, n, survivors):
        offspring = []
        while len(offspring) < n:
            indices = np.random.choice(len(survivors), 2, replace=False)
            parent1, parent2 = survivors[indices[0]], survivors[indices[1]]

            child = self.crossover(parent1, parent2)
            child = self.mutate_gene_array(child)
            offspring.append(child)
        
        return np.array(offspring)

    def best_patch(self, base_img, completion = 1, verbose=True):
        pop = self.initialize_patch_pop(base_img, completion)

        best_mse = float('inf')
        no_improve_count = 0

        for generation in range(self.parameters.max_patch_generations):

            with ThreadPoolExecutor() as executor:
                fitness_scores = list(executor.map(lambda p: self.evaluate_patch(base_img, p), pop))
            fitness_scores = np.array(fitness_scores)

            # sort by fitness
            sorted_indices = np.argsort(fitness_scores)
            pop = pop[sorted_indices]
            fitness_scores = fitness_scores[sorted_indices]

            current_best_mse = fitness_scores[0]

            # early stop check
            if generation >= self.parameters.min_generations:
                if best_mse - current_best_mse < self.parameters.early_stop_eps:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    best_mse = current_best_mse

                if (no_improve_count >= self.parameters.early_stop_patience):
                    if verbose:
                        print(f"    Patch {round(completion * self.parameters.final_patch_pop_size)}: Early stopping at generation {generation+1}. Best Score (MSE): {current_best_mse:.6f}")
                    break
            else:
                best_mse = current_best_mse  # initialize after enough generations

            # keep top survivors
            survivors_count = int(len(pop) * self.parameters.survival_proportion)
            survivors = pop[:survivors_count]

            # reproduce to refill the population
            offspring = self.get_offspring(len(pop) - survivors_count, survivors = survivors)
            pop = np.concatenate([survivors, offspring])

            if verbose:
                print(f"    Patch {round(completion * self.parameters.final_patch_pop_size)}, Generation {generation+1}: Best Score (MSE): {current_best_mse:.6f}")

        # return the best individual after final generation or early stop
        final_fitness = [self.evaluate_patch(base_img, p) for p in pop]
        best_idx = np.argmin(final_fitness)
        return pop[best_idx]
    
    def fancy_patch_string(self, patch_arr):
        return f"Added patch #{int(patch_arr[0])} @ ({round(patch_arr[2], 2)}, {round(patch_arr[1], 2)})!"
    
    def render_animation(self, filepath, fps=12):
        if not self.parameters.make_animation:
            raise ValueError("make_animation is set to false.")
        
        if len(self.video) == 0:
            raise ValueError("Video list is empty.")

        height, width, channels = self.video[0].shape

        # define the codec (use 'mp4v' for .mp4 and 'XVID' for .avi)
        ext = filepath.split('.')[-1].lower()
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise ValueError(f"Unsupported file extension: .{ext}")

        # create the video writer object
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        for frame in self.video:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames must have the same dimensions.")
            if frame.dtype != np.uint8:
                raise ValueError("All frames must be of type np.uint8.")
            out.write(frame)

        out.release()
        print(f"Animation saved to {filepath}!")

    def run(self, notebook = False, verbose = True):
        def show_frame(img):
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            display(plt.gcf())
            clear_output(wait=True)

        working_img = self.init_working_img()

        if notebook:
            show_frame(working_img)
        if self.parameters.make_animation:
            self.video.append(working_img)

        for i in range(self.parameters.final_patch_pop_size):
            best_patch_arr = self.best_patch(working_img, completion = i / self.parameters.final_patch_pop_size, verbose = (verbose and not notebook))
            working_img = self.render_patch(working_img, best_patch_arr)

            if notebook:
                show_frame(working_img)
            else:
                print(self.fancy_patch_string(best_patch_arr))
            
            if self.parameters.make_animation:
                self.video.append(working_img)

        return working_img