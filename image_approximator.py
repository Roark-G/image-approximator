import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import json
from PIL import Image

try:  # works both inside a package and as loose files side by side
    from .progress_bar import ProgressBar
except ImportError:
    from progress_bar import ProgressBar

'''
# PATCH ARRAY (always 9 elements)
# [patch_id (index into self.patch_files),
#  y ([0-1) : 0 is top 1 is bottom),
#  x ([0-1) : 0 is left 1 is right),
#  angle ([0-1) : wraps around, 0.5 is 180 degrees),
#  size (patch height as a fraction of the canvas height),
#  recolor_r, recolor_g, recolor_b ([0-1]),
#  recolor_intensity ([0-1] : 0 is no change, 1 is full recolor)]
#
# The stored patches (self.patches) are always "ready to draw": rotation, size and recolor
# intensity already reflect the settings of the run that made them, so rendering needs no parameters.
'''
ID, Y, X, ANGLE, SIZE, R, G, B, INTENSITY = range(9)
NUM_GENES = 9
JSON_VERSION = 2


class RunParameters():
    """Search settings for run(). Rendering and exporting take their settings as method arguments."""
    def __init__(self):
        self.start_mode = "average"            # first run only: "white", "average" (of the target), or an image path

        # PATCH PARAMETERS
        self.allow_angle = True                # allow patch rotation
        self.allow_recolor = True              # allow patch recoloring
        self.max_recolor_intensity = 1.0       # maximum recolor intensity
        self.perfect_color = True              # use perfect color matching

        self.allow_resize = True               # allow patch resizing
        self.start_min_patch_size = 0.1        # size range for a run's first patch...
        self.start_max_patch_size = 1.0
        self.min_patch_size = 0.0              # ...shrinking linearly to this range by its last patch
        self.max_patch_size = 0.2

        # RUN PARAMETERS
        self.find_patch_pop_size = 100         # searching patch population size
        self.search_resolution = 256           # longest edge of the search canvas (None: search at the canvas size)
        self.patch_init_pixel_size = 1         # initial patch size in pixels
        self.survival_proportion = 0.25        # proportion of patches that survive each generation

        self.early_stop_eps = 1e-3             # minimum improvement to continue searching (for optimization)
        self.early_stop_patience = 15          # number of generations without improvement before stopping
        self.min_generations = 10              # minimum number of generations to run
        self.max_patch_generations = 100       # maximum number of generations to run

        # MUTATION PARAMETERS
        self.mutation_chance = 0.15            # chance of mutating a gene
        self.mutation_std = 0.05               # standard deviation of gene mutations

        # NOTEBOOK PREVIEW
        self.preview_longest_edge = 400        # longest edge of the live preview in pixels (None: show the search canvas)
        self.preview_every = 1                 # refresh the preview every n patches


class Canvas():
    """One drawing surface: its size and, when searching, the target to match."""
    def __init__(self, width, height, target_img=None):
        self.width, self.height = width, height
        self.target_img = target_img
        self.target_float = target_img.astype(np.float32) if target_img is not None else None
        self.num_values = height * width * 3  # MSE denominator


class ImageApproximator():
    """
    Holds an approximation: the canvas size, the background, the placed patches, and every patch image they use
    """
    def __init__(self):
        self.width = self.height = None     # canvas size: set by the first run() or load_json()
        self.background = (255, 255, 255)   # BGR color tuple, or a BGR image
        self.background_path = None         # path of the background image, if it is one
        self.patches = []                   # placed patches, in drawing order (one 9-element array each)
        self.patch_files = []               # patch image paths; a patch's ID is its index here
        self.raw_patches = []               # the loaded patch images (BGRA), parallel to patch_files

        self._scaled_patch_cache = {}       # (patch id, height) -> image
        # set only while run() is going
        self._p = self._search = self._executor = None

    def reset(self):
        """Clear everything, including the canvas size, so the next run() starts fresh."""
        self.__init__()
        return self

    # LOADING
    def get_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image at path: {path}")
        return img

    def to_bgra(self, img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def read_patch(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype != np.uint8:
            img = (img / 257).astype(np.uint8)  # 16-bit PNG -> 8-bit
        return self.to_bgra(img)

    def add_patch_file(self, filepath):
        """Add one patch image to the library (once per file). Returns its patch ID, or None if unreadable."""
        key = os.path.abspath(filepath)
        for i, existing in enumerate(self.patch_files):
            if os.path.abspath(existing) == key:
                return i
        img = self.read_patch(filepath)
        if img is None:
            print(f"Warning: Failed to load {filepath}")
            return None
        self.patch_files.append(filepath)
        self.raw_patches.append(img)
        return len(self.patch_files) - 1

    def add_patch_library(self, patches_dir):
        """Add every PNG in a folder (or list of folders). Returns the patch IDs of that library."""
        dirs = [patches_dir] if isinstance(patches_dir, str) else list(patches_dir)
        ids = []
        for d in dirs:
            for filename in sorted(os.listdir(d)):
                if filename.lower().endswith(".png"):
                    i = self.add_patch_file(os.path.join(d, filename))
                    if i is not None and i not in ids:
                        ids.append(i)
        if not ids:
            raise ValueError(f"No PNG patches found in: {patches_dir}")
        return ids

    # ------------------------------------------------------------------ canvases

    def scaled_patch(self, patch_id, height):
        """A patch image scaled to a canvas height (cached)."""
        key = (patch_id, height)
        if key not in self._scaled_patch_cache:
            patch = self.raw_patches[patch_id]
            h, w = patch.shape[:2]
            new_w = max(1, round(height * w / float(h)))
            if (height, new_w) == (h, w):
                scaled = patch
            else:
                interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
                scaled = cv2.resize(patch, (new_w, height), interpolation=interp)
            self._scaled_patch_cache[key] = scaled
        return self._scaled_patch_cache[key]

    def output_size(self, longest_edge=None):
        """(width, height) for rendering, preserving aspect ratio. longest_edge=None keeps the canvas size."""
        if self.width is None:
            raise ValueError("Nothing to render yet: call run() or load_json() first.")
        if longest_edge is None:
            return self.width, self.height
        scale = longest_edge / float(max(self.width, self.height))
        return max(1, round(self.width * scale)), max(1, round(self.height * scale))

    def background_image(self, width, height):
        if isinstance(self.background, np.ndarray):
            h, w = self.background.shape[:2]
            if (w, h) == (width, height):
                return self.background.copy()
            interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
            return cv2.resize(self.background, (width, height), interpolation=interp)
        return np.full((height, width, 3), self.background, dtype=np.uint8)

    def background_string(self):
        """The background as "#rrggbb", or "image"."""
        if isinstance(self.background, np.ndarray):
            return "image"
        b, g, r = self.background
        return f"#{r:02x}{g:02x}{b:02x}"

    def set_background(self, mode, target_img):
        if mode == "white":
            self.background, self.background_path = (255, 255, 255), None
        elif mode == "average":
            self.background = tuple(int(v) for v in target_img.mean(axis=(0, 1)).astype(np.uint8))
            self.background_path = None
        elif isinstance(mode, str) and os.path.isfile(mode):
            self.background, self.background_path = self.get_image(mode), mode
        else:
            raise ValueError(f"Invalid start_mode: {mode}. Choose 'white', 'average', or a valid image path.")

    # ------------------------------------------------------------------ drawing

    def recolor(self, image_bgra, patch):
        """Blend every visible pixel of the patch toward its color by its intensity."""
        intensity = float(np.clip(patch[INTENSITY], 0.0, 1.0))
        if intensity == 0:
            return image_bgra

        rgb = patch[R:B + 1] * 255
        target_bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
        bgr = image_bgra[:, :, :3].astype(np.float32)
        visible = image_bgra[:, :, 3:4] > 0

        blended = np.where(visible, (1 - intensity) * bgr + intensity * target_bgr, bgr)

        recolored = image_bgra.copy()
        recolored[:, :, :3] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        return recolored

    def transform_patch(self, patch, canvas):
        """Resize and rotate the patch image for this canvas (no recolor). Returns BGRA uint8."""
        patch_img = self.scaled_patch(int(patch[ID]), canvas.height)

        h, w = patch_img.shape[:2]
        target_height = max(1, int(patch[SIZE] * canvas.height + 1e-6))  # prevent zero height
        new_w = max(1, int(target_height * w / h))                        # prevent zero width
        if (target_height, new_w) != (h, w):
            patch_img = cv2.resize(patch_img, (new_w, target_height), interpolation=cv2.INTER_AREA)

        angle = patch[ANGLE] % 1.0
        if angle != 0:
            h, w = patch_img.shape[:2]
            center = (w / 2, h / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle * 360, 1.0)
            # expand the canvas so rotated corners aren't clipped
            cos, sin = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
            new_w = int(np.ceil(h * sin + w * cos))
            new_h = int(np.ceil(h * cos + w * sin))
            rot_mat[0, 2] += new_w / 2 - center[0]
            rot_mat[1, 2] += new_h / 2 - center[1]
            patch_img = cv2.warpAffine(patch_img, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        return patch_img

    def placement(self, patch_img, patch, canvas):
        """Return (y0, y1, x0, x1, py0, py1, px0, px1) for the visible overlap, or None if off-canvas."""
        patch_h, patch_w = patch_img.shape[:2]

        x_pos = int(patch[X] * canvas.width) - patch_w // 2
        y_pos = int(patch[Y] * canvas.height) - patch_h // 2

        x0, y0 = max(x_pos, 0), max(y_pos, 0)
        x1, y1 = min(x_pos + patch_w, canvas.width), min(y_pos + patch_h, canvas.height)
        if x1 <= x0 or y1 <= y0:
            return None

        px0, py0 = x0 - x_pos, y0 - y_pos
        return (y0, y1, x0, x1, py0, py0 + (y1 - y0), px0, px0 + (x1 - x0))

    def blend_roi(self, base_roi, patch_img, place):
        """Alpha-blend the visible part of patch_img over base_roi. Returns a new uint8 ROI."""
        _, _, _, _, py0, py1, px0, px1 = place
        crop = patch_img[py0:py1, px0:px1]
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        blended = alpha * crop[:, :, :3].astype(np.float32) + (1 - alpha) * base_roi.astype(np.float32)
        return np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    def render_patch(self, base_img, patch, canvas):
        """Draw one patch onto a copy of base_img (which must match the canvas size)."""
        base_img = base_img.copy()
        patch_img = self.transform_patch(patch, canvas)
        place = self.placement(patch_img, patch, canvas)
        if place is None:
            return base_img
        y0, y1, x0, x1 = place[:4]
        base_img[y0:y1, x0:x1] = self.blend_roi(base_img[y0:y1, x0:x1], self.recolor(patch_img, patch), place)
        return base_img

    def render_on(self, canvas, count=None):
        """Background plus the first `count` patches (all if None) on a canvas."""
        img = self.background_image(canvas.width, canvas.height)
        for patch in self.patches[:count]:
            img = self.render_patch(img, patch, canvas)
        return img

    def render(self, longest_edge=None):
        """Render the stored result. longest_edge=None renders at the canvas size."""
        width, height = self.output_size(longest_edge)
        return self.render_on(Canvas(width, height))

    # ------------------------------------------------------------------ exporting

    def export_img(self, filepath, dpi=300, longest_edge=1400):
        """
        Render and save the result with DPI metadata. Aspect ratio is preserved;
        longest_edge=None saves at the canvas size. Print size = longest_edge / dpi inches.
        Returns the rendered BGR image.
        """
        img = self.render(longest_edge)
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(filepath, dpi=(dpi, dpi))
        h, w = img.shape[:2]
        print(f"Image saved to {filepath} ({w}x{h} px at {dpi} DPI = {w / dpi:.2f}x{h / dpi:.2f} in)!")
        return img

    def export_animation(self, filepath, fps=12, longest_edge=1400):
        """Render the result one patch per frame and save it as .mp4 or .avi."""
        if len(self.patches) == 0:
            raise ValueError("No patches to animate. Call run() or load_json() first.")

        ext = filepath.split('.')[-1].lower()
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise ValueError(f"Unsupported file extension: .{ext}")

        width, height = self.output_size(longest_edge)
        canvas = Canvas(width, height)
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        frame = self.background_image(width, height)
        out.write(frame)
        for patch in self.patches:
            frame = self.render_patch(frame, patch, canvas)
            out.write(frame)
        out.release()
        print(f"Animation saved to {filepath}!")

    def get_patch_array(self):
        """The patches as an (n, 9) array, in drawing order."""
        return np.array(self.patches).reshape(-1, NUM_GENES)

    def export_json(self, filepath):
        """
        Write the result as JSON:

        {
          "version": JSON_VERSION,
          "canvas": {"width", "height", "background"},   # background: "#rrggbb", or "image" plus "start_image"
          "patch_files": ["patches/a.png", ...],         # paths as given to run(); index = patch id
          "patches": [                                   # drawing order (first to last)
            {"patch", "x", "y", "angle", "size", "color": [r, g, b], "intensity",
             "angle_degrees", "w_px", "h_px"}
          ]
        }
        """
        if len(self.patches) == 0:
            raise ValueError("No patches to export. Call run() or load_json() first.")

        rnd = lambda v: round(float(v), 6)
        canvas = {"width": self.width, "height": self.height, "background": self.background_string()}
        if canvas["background"] == "image":
            canvas["start_image"] = self.background_path

        patches = []
        for p in self.patches:
            rh, rw = self.raw_patches[int(p[ID])].shape[:2]
            h_px = max(1, int(p[SIZE] * self.height + 1e-6))
            patches.append({
                "patch": int(p[ID]),
                "x": rnd(p[X]),
                "y": rnd(p[Y]),
                "angle": rnd(p[ANGLE]),
                "size": rnd(p[SIZE]),
                "color": [rnd(p[R]), rnd(p[G]), rnd(p[B])],
                "intensity": rnd(p[INTENSITY]),
                "angle_degrees": rnd((p[ANGLE] % 1.0) * 360),
                "w_px": max(1, int(h_px * rw / rh)),
                "h_px": h_px,
            })

        data = {"version": JSON_VERSION, "canvas": canvas,
                "patch_files": [f.replace(os.sep, "/") for f in self.patch_files], "patches": patches}
        with open(filepath, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Patches saved to {filepath}!")

    def find_patch_file(self, name, json_dir, patch_dirs):
        """Locate a patch file listed in a JSON: as written, next to the JSON, or by filename in patch_dirs."""
        candidates = [name, os.path.join(json_dir, name)]
        candidates += [os.path.join(d, os.path.basename(name)) for d in patch_dirs]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def load_json(self, filepath, patch_dirs=None):
        """
        Replace everything with the result in a JSON from export_json (older exports also load)
        """
        with open(filepath) as f:
            data = json.load(f)
        version = data.get("version", 1)
        canvas = data["canvas"]
        patch_dirs = [patch_dirs] if isinstance(patch_dirs, str) else list(patch_dirs or [])

        # find every patch file the drawing uses before changing anything
        json_dir = os.path.dirname(os.path.abspath(filepath))
        used = sorted({p["patch"] for p in data["patches"]})
        paths, missing = {}, []
        for i in used:
            path = self.find_patch_file(data["patch_files"][i], json_dir, patch_dirs)
            if path is None:
                missing.append(data["patch_files"][i])
            paths[i] = path
        if missing:
            raise ValueError(f"Patch files used in {filepath} were not found (pass patch_dirs=...): {missing}")

        background, background_path = (255, 255, 255), None
        if canvas["background"] == "image":
            background_path = canvas.get("start_image")
            if not background_path or not os.path.isfile(background_path):
                raise ValueError(f"Start image for this JSON not found: {background_path}")
            background = self.get_image(background_path)
        else:
            hex_color = canvas["background"].lstrip("#")
            r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            background = (b, g, r)

        self.reset()
        self.width, self.height = canvas["width"], canvas["height"]
        self.background, self.background_path = background, background_path
        id_map = {}
        for i in used:
            id_map[i] = self.add_patch_file(paths[i])
            if id_map[i] is None:
                raise ValueError(f"Failed to load patch file: {paths[i]}")

        for p in data["patches"]:
            patch = np.zeros(NUM_GENES)
            patch[ID] = id_map[p["patch"]]
            patch[Y], patch[X] = p["y"], p["x"]
            patch[R:B + 1] = p["color"]
            if version >= 2:
                patch[ANGLE], patch[SIZE], patch[INTENSITY] = p["angle"], p["size"], p["intensity"]
            else:  # version 1 stored raw search genes; use its ready-to-draw fields
                patch[ANGLE] = p["angle_degrees"] / 360.0
                patch[SIZE] = p["h_px"] / canvas["height"]
                patch[INTENSITY] = p.get("applied_intensity", p["intensity"])
            self.patches.append(patch)

        print(f"Loaded {len(self.patches)} patches from {filepath}.")
        return self

    # SEARCHING
    def make_search_canvas(self, target_img, res):
        """The run's target, resized to the canvas shape at search resolution."""
        width, height = self.output_size(res)
        if res is not None and res >= max(self.width, self.height):
            width, height = self.width, self.height
        h, w = target_img.shape[:2]
        if (w, h) != (width, height):
            interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
            target_img = cv2.resize(target_img, (width, height), interpolation=interp)
        return Canvas(width, height, target_img)

    def get_active_genes(self):
        """Indices of the continuous genes the GA is allowed to search over."""
        p = self._p
        genes = [Y, X]
        if p.allow_angle:
            genes.append(ANGLE)
        if p.allow_resize:
            genes.append(SIZE)
        if p.allow_recolor and not p.perfect_color:
            genes.extend([R, G, B, INTENSITY])
        return genes

    def random_patch_id(self):
        return self._library[np.random.randint(0, len(self._library))]

    def to_patch(self, genes):
        """Turn search genes into a ready-to-draw patch using the run's settings."""
        p = self._p
        patch = genes.copy()
        patch[ANGLE] = genes[ANGLE] % 1.0 if p.allow_angle else 0.0
        if not p.allow_resize:
            patch[SIZE] = self._native_sizes[int(genes[ID])]
        if p.allow_recolor:
            patch[INTENSITY] = float(np.clip(genes[INTENSITY] * p.max_recolor_intensity, 0.0, 1.0))
        else:
            patch[R:B + 1] = 0.0
            patch[INTENSITY] = 0.0
        return patch

    def size_bounds(self):
        p, t = self._p, self._completion
        lo = p.start_min_patch_size + (p.min_patch_size - p.start_min_patch_size) * t
        hi = p.start_max_patch_size + (p.max_patch_size - p.start_max_patch_size) * t
        lo, hi = float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))
        return lo, max(lo, hi)

    def make_random_patch(self, position=None):
        p = self._p
        genes = np.zeros(NUM_GENES)
        genes[ID] = self.random_patch_id()

        if position is not None:
            genes[Y], genes[X] = position
        else:
            genes[Y], genes[X] = np.random.uniform(), np.random.uniform()

        genes[ANGLE] = np.random.uniform() if p.allow_angle else 0.0

        if p.allow_resize:
            lo, hi = self.size_bounds()
            genes[SIZE] = np.random.uniform(lo, hi)

        if p.allow_recolor:
            if p.perfect_color:
                genes[INTENSITY] = 1.0  # color is solved per candidate in evaluate_patch
            else:
                genes[R:B + 1] = np.random.uniform(size=3)
                genes[INTENSITY] = 1 - np.random.uniform()

        return genes

    def sample_positions(self, error_heatmap, n):
        """Sample n (y, x) positions in [0, 1) weighted by the error heatmap."""
        prob_map = error_heatmap.flatten().astype(np.float64)
        prob_map /= prob_map.sum()
        idx = np.random.choice(len(prob_map), size=n, p=prob_map)
        h, w = error_heatmap.shape
        y_idx, x_idx = np.divmod(idx, w)
        return np.stack([(y_idx + 0.5) / h, (x_idx + 0.5) / w], axis=1)

    def pixelate(self, img, pixel_size):
        if pixel_size <= 1:
            return img
        h, w = img.shape[:2]
        temp = cv2.resize(img, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    def make_error_heatmap(self, img):
        abs_error = np.abs(img.astype(np.float32) - self._search.target_float)
        abs_error = np.linalg.norm(abs_error, axis=2)  # shape: (H, W)
        abs_error += 1e-6
        abs_error /= abs_error.sum()
        return self.pixelate(abs_error, self._p.patch_init_pixel_size)

    def initialize_patch_pop(self, img):
        positions = self.sample_positions(self.make_error_heatmap(img), self._p.find_patch_pop_size)
        return np.array([self.make_random_patch(position=pos) for pos in positions])

    def crossover(self, parent1, parent2):
        """Uniform crossover: each gene comes whole from one parent (no averaging)."""
        child = parent1.copy()
        if np.random.rand() < 0.5:
            child[ID] = parent2[ID]
        for i in self._active_genes:
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
        return child

    def mutate_gene_array(self, gene_array):
        """
        Gaussian noise on the active genes. Angle wraps around; other genes reflect to stay
        within [0, 1]. The patch ID is mutated separately, within the run's patch library.
        """
        p = self._p
        mutated = gene_array.copy()

        if np.random.rand() < p.mutation_chance:
            mutated[ID] = self.random_patch_id()

        for i in self._active_genes:
            if np.random.rand() < p.mutation_chance:
                mutated[i] += np.random.normal(0, p.mutation_std)
                if i == ANGLE:
                    mutated[i] %= 1.0
                else:
                    lo, hi = self.size_bounds() if i == SIZE else (0.0, 1.0)
                    if mutated[i] < lo:
                        mutated[i] = 2 * lo - mutated[i]
                    if mutated[i] > hi:
                        mutated[i] = 2 * hi - mutated[i]
                    mutated[i] = max(lo, min(hi, mutated[i]))  # clamp just in case

        return mutated

    def squared_error(self, img, target_float):
        return float(np.sum((img.astype(np.float32) - target_float) ** 2, dtype=np.float64))

    def solve_perfect_color(self, patch_img, base_img, place, canvas, i):
        """
        Least-squares optimal flat recolor for this patch at this placement.
        Composite pixel = a*((1-i)*p + i*c) + (1-a)*base, so per channel
        c = sum(a*i * (target - (1-a)*base - a*(1-i)*p)) / sum((a*i)^2).
        Returns RGB in [0, 1].
        """
        y0, y1, x0, x1, py0, py1, px0, px1 = place
        crop = patch_img[py0:py1, px0:px1]
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0

        patch_bgr = crop[:, :, :3].astype(np.float32)
        base = base_img[y0:y1, x0:x1].astype(np.float32)
        target = canvas.target_float[y0:y1, x0:x1]

        weight = alpha * i
        denom = float((weight ** 2).sum())
        if denom < 1e-8:
            return np.zeros(3)

        residual = target - (1 - alpha) * base - alpha * (1 - i) * patch_bgr
        color_bgr = np.clip((weight * residual).sum(axis=(0, 1)) / denom, 0, 255)
        return color_bgr[::-1] / 255.0  # BGR -> RGB

    def evaluate_patch(self, base_img, genes, total_sq_err):
        """
        Score candidate genes on the search canvas. Only the patch's bounding box is re-scored:
        mse = (total_error - old_roi_error + new_roi_error) / num_values.
        Returns (mse, ready-to-draw patch).
        """
        canvas = self._search
        patch = self.to_patch(genes)
        patch_img = self.transform_patch(patch, canvas)
        place = self.placement(patch_img, patch, canvas)

        if place is None:
            return total_sq_err / canvas.num_values, patch

        if self._p.allow_recolor and self._p.perfect_color:
            patch[R:B + 1] = self.solve_perfect_color(patch_img, base_img, place, canvas, patch[INTENSITY])

        y0, y1, x0, x1 = place[:4]
        base_roi = base_img[y0:y1, x0:x1]
        target_roi = canvas.target_float[y0:y1, x0:x1]
        new_roi = self.blend_roi(base_roi, self.recolor(patch_img, patch), place)

        old_err = self.squared_error(base_roi, target_roi)
        new_err = self.squared_error(new_roi, target_roi)
        return (total_sq_err - old_err + new_err) / canvas.num_values, patch

    def evaluate_pop(self, base_img, pop, total_sq_err):
        results = list(self._executor.map(lambda g: self.evaluate_patch(base_img, g, total_sq_err), pop))
        fitness_scores = np.array([r[0] for r in results])
        patches = np.array([r[1] for r in results])
        return fitness_scores, patches

    def get_offspring(self, n, survivors):
        offspring = []
        while len(offspring) < n:
            i1, i2 = np.random.choice(len(survivors), 2, replace=False)
            child = self.crossover(survivors[i1], survivors[i2])
            offspring.append(self.mutate_gene_array(child))
        return np.array(offspring)

    def best_patch(self, base_img, label=None, verbose=True):
        """Find the best next patch for the search-resolution image. Returns a ready-to-draw patch."""
        p = self._p
        pop = self.initialize_patch_pop(base_img)
        total_sq_err = self.squared_error(base_img, self._search.target_float)

        best_mse = float('inf')
        no_improve_count = 0

        for generation in range(p.max_patch_generations):
            fitness_scores, patches = self.evaluate_pop(base_img, pop, total_sq_err)

            order = np.argsort(fitness_scores)
            pop, patches, fitness_scores = pop[order], patches[order], fitness_scores[order]
            current_best_mse = fitness_scores[0]

            # early stop check
            if generation >= p.min_generations:
                if best_mse - current_best_mse < p.early_stop_eps:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    best_mse = current_best_mse

                if no_improve_count >= p.early_stop_patience:
                    if verbose:
                        print(f"    Patch {label}: Early stopping at generation {generation+1}. Best Score (MSE): {current_best_mse:.6f}")
                    return patches[0]
            else:
                best_mse = current_best_mse  # initialize after enough generations

            # keep top survivors (at least 2 so crossover has two parents)
            survivors_count = max(2, int(len(pop) * p.survival_proportion))
            survivors = pop[:survivors_count]

            offspring = self.get_offspring(len(pop) - survivors_count, survivors=survivors)
            pop = np.concatenate([survivors, offspring]) if len(offspring) else survivors

            if verbose:
                print(f"    Patch {label}, Generation {generation+1}: Best Score (MSE): {current_best_mse:.6f}")

        fitness_scores, patches = self.evaluate_pop(base_img, pop, total_sq_err)
        return patches[np.argmin(fitness_scores)]

    def fancy_patch_string(self, patch, n=None, total=None):
        s = f"Added patch #{int(patch[ID])} @ ({round(patch[X], 2)}, {round(patch[Y], 2)})!"
        if n is not None and total:
            s += f" [{n}/{total} patches | {n / total * 100:.1f}%]"
        return s

    def run(self, target, patches_dir, n_patches=200, parameters=None, notebook=False, verbose=True):
        """
        Add n_patches patches from patches_dir (a folder or list of folders) that move the current
        image toward target (an image path or a BGR array), on top of whatever is already placed.

        The first run sets the canvas size from its target and the background from start_mode.
        Later targets are resized to that canvas. Nothing is saved; use the export methods. Returns self.
        """
        p = parameters if parameters is not None else RunParameters()
        if p.perfect_color and not p.allow_recolor:
            print("Warning: perfect_color has no effect when allow_recolor is False.")

        target_img = self.get_image(target) if isinstance(target, str) else np.asarray(target, dtype=np.uint8)
        th, tw = target_img.shape[:2]
        library = self.add_patch_library(patches_dir)

        if self.width is None:  # first run: this target defines the canvas
            self.width, self.height = tw, th
            self.patches = []
            self.set_background(p.start_mode, target_img)
        elif abs(tw / th - self.width / self.height) > 0.01:
            print(f"Warning: target is {tw}x{th} but the canvas is {self.width}x{self.height}; the target will be stretched to fit.")

        preview = bar = None
        show_bar = not verbose and not notebook  # quiet terminal runs get a one-line progress bar
        if notebook:
            from IPython.display import display
            to_pil = lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        self._p = p
        try:
            self._search = self.make_search_canvas(target_img, p.search_resolution)
            self._library = library
            # with resizing off, patches keep their pixel size relative to this run's target
            self._native_sizes = {i: self.raw_patches[i].shape[0] / float(th) for i in library}
            self._active_genes = self.get_active_genes()

            if verbose and not notebook:
                print(f"Searching at {self._search.width}x{self._search.height} with {len(library)} patch images.")

            search_img = self.render_on(self._search)  # background plus patches from earlier runs
            if notebook:
                if p.preview_longest_edge is None:
                    preview_canvas, preview = self._search, search_img
                else:
                    preview_canvas = Canvas(*self.output_size(p.preview_longest_edge))
                    preview = self.render_on(preview_canvas)
                handle = display(to_pil(preview), display_id=True)  # updated in place: no flicker

            with ThreadPoolExecutor() as executor:
                self._executor = executor
                if show_bar:
                    bar = ProgressBar(n_patches, label="Approximating", unit="patch")
                for i in range(n_patches):
                    self._completion = i / n_patches
                    patch = self.best_patch(search_img, label=len(self.patches) + 1,
                                            verbose=(verbose and not notebook))
                    self.patches.append(patch)
                    search_img = self.render_patch(search_img, patch, self._search)

                    if notebook:
                        preview = search_img if preview_canvas is self._search else \
                            self.render_patch(preview, patch, preview_canvas)
                        if (i + 1) % max(1, p.preview_every) == 0 or i == n_patches - 1:
                            handle.update(to_pil(preview))
                    elif verbose:
                        print(self.fancy_patch_string(patch, i + 1, n_patches))
                    else:
                        mse = self.squared_error(search_img, self._search.target_float) / self._search.num_values
                        bar.update(postfix=f"MSE {mse:.2f}")
        finally:
            if bar is not None:
                bar.close()
            self._p = self._search = self._executor = self._library = None

        return self