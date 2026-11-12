import shutil
import sys
import time


class ProgressBar():
    """
    A one-line terminal progress bar, redrawn in place:
    """
    FULL = "█"
    PARTIALS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]  # 1/8 .. 7/8 of a cell
    EMPTY = "·"

    def __init__(self, total, label="Progress", unit="it", stream=None, max_bar_width=40):
        self.total = max(0, int(total))
        self.label = label
        self.unit = unit
        self.stream = stream if stream is not None else sys.stdout
        self.max_bar_width = max_bar_width
        self.count = 0
        self.postfix = ""
        self.start = time.perf_counter()
        self._last_len = 0
        self._closed = False
        self._draw()

    # ------------------------------------------------------------------ public

    def update(self, n=1, postfix=None):
        """Advance by n steps; postfix is extra text shown at the end of the line (e.g. a score)."""
        self.count = min(self.total, self.count + n)
        if postfix is not None:
            self.postfix = postfix
        self._draw()

    def close(self):
        """Draw the final state and move to a new line (safe to call more than once)."""
        if self._closed:
            return
        self._closed = True
        self._draw(final=True)
        self.stream.write("\n")
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ drawing

    @staticmethod
    def format_time(seconds):
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def bar_string(self, fraction, width):
        eighths = int(fraction * width * 8)
        full, part = divmod(eighths, 8)
        bar = self.FULL * full + (self.PARTIALS[part] if full < width else "")
        return bar + self.EMPTY * (width - len(bar))

    def rate_string(self, elapsed):
        if self.count == 0 or elapsed <= 0:
            return f"-- {self.unit}/s"
        rate = self.count / elapsed
        if rate >= 1:
            return f"{rate:.2f} {self.unit}/s"
        singular = self.unit[:-1] if self.unit.endswith("s") and len(self.unit) > 1 else self.unit
        return f"{1 / rate:.2f} s/{singular}"  # slow steps read better as seconds per step

    def _draw(self, final=False):
        elapsed = time.perf_counter() - self.start
        fraction = self.count / self.total if self.total else 1.0

        if final or self.count >= self.total:
            timing = self.format_time(elapsed)
        elif self.count:
            timing = f"{self.format_time(elapsed)}<{self.format_time(elapsed / self.count * (self.total - self.count))}"
        else:
            timing = f"{self.format_time(elapsed)}<--:--"

        digits = len(str(self.total))
        tail = f"│ {self.count:>{digits}}/{self.total} {fraction * 100:5.1f}% │ {timing} │ {self.rate_string(elapsed)}"
        if self.postfix:
            tail += f" │ {self.postfix}"

        # fit the line to the terminal: a wrapped line can't be redrawn with \r
        columns = shutil.get_terminal_size((100, 20)).columns
        head = f"{self.label} │"
        width = max(10, min(self.max_bar_width, columns - len(head) - len(tail) - 2))
        line = f"{head}{self.bar_string(fraction, width)}{tail}"[:max(1, columns - 1)]

        pad = " " * max(0, self._last_len - len(line))  # wipe leftovers from a longer previous line
        self.stream.write("\r" + line + pad)
        self.stream.flush()
        self._last_len = len(line)