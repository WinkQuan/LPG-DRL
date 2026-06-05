#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Plot reward and success-rate training curves exported from wandb."""

from __future__ import print_function

import argparse
import csv
import html
import re
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REWARD_CSV = SCRIPT_DIR / "reward.csv"
DEFAULT_SUCCESS_CSV = SCRIPT_DIR / "success  rate.csv"
DEFAULT_REWARD_OUTPUT = SCRIPT_DIR / "reward_curve.pdf"
DEFAULT_SUCCESS_OUTPUT = SCRIPT_DIR / "success_rate_curve.pdf"
DEFAULT_MAX_STEP = 2000.0
DEFAULT_SMOOTH_WINDOW = 100
DEFAULT_FIGURE_WIDTH_IN = 3.5
DEFAULT_FIGURE_HEIGHT_IN = 2.45
SVG_UNITS_PER_IN = 144
SUCCESS_RATE_BOUNDS = (0.0, 1.0)
SVG_AXIS_LABEL_OFFSET = 22
PDF_AXIS_LABEL_OFFSET = 15.5

ALGORITHM_ORDER = ["LPG-DRL", "NPE-DRL", "D3QN", "D3QN-LfD", "BC", "DAgger", "APF"]
EXCLUDED_ALGORITHMS = {"PPO"}
COLORS = {
    "LPG-DRL": "#d62728",
    "NPE-DRL": "#ff7f0e",
    "D3QN": "#1f77b4",
    "D3QN-LfD": "#1f77b4",
    "BC": "#2ca02c",
    "DAgger": "#8c564b",
    "APF": "#7f7f7f",
}
LEGACY_ALGORITHM_LABELS = {
    "PAL-DRL": "LPG-DRL",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot wandb reward and success-rate training curves.")
    parser.add_argument("--reward-csv", type=Path, default=DEFAULT_REWARD_CSV, help="Path to wandb reward CSV.")
    parser.add_argument(
        "--success-csv",
        type=Path,
        default=DEFAULT_SUCCESS_CSV,
        help="Path to wandb success-rate CSV.",
    )
    parser.add_argument("--reward-output", type=Path, default=DEFAULT_REWARD_OUTPUT, help="Reward curve image path.")
    parser.add_argument(
        "--success-output",
        type=Path,
        default=DEFAULT_SUCCESS_OUTPUT,
        help="Success-rate curve image path.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=DEFAULT_MAX_STEP,
        help="Maximum Step/Episode to plot. Later rows are discarded before smoothing.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Trailing moving-average window. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--figure-width-in",
        type=float,
        default=DEFAULT_FIGURE_WIDTH_IN,
        help="Figure width in inches. IEEE single-column width is about 3.5 inches.",
    )
    parser.add_argument(
        "--figure-height-in",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT_IN,
        help="Figure height in inches.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def algorithm_from_column(column, metric_name):
    run_name = column.split(f" - {metric_name}", 1)[0].strip()
    match = re.match(r"(?P<algorithm>.+?)_\d{4}-\d{2}-\d{2}", run_name)
    if match:
        algorithm = match.group("algorithm")
    else:
        algorithm = run_name
    return LEGACY_ALGORITHM_LABELS.get(algorithm, algorithm)


def unique_label(label, existing_labels):
    if label not in existing_labels:
        return label

    index = 2
    while f"{label}_{index}" in existing_labels:
        index += 1
    return f"{label}_{index}"


def metric_columns(fieldnames, metric_name):
    columns = []
    for column in fieldnames:
        if column == "Step":
            continue
        if column.endswith("__MIN") or column.endswith("__MAX"):
            continue
        if f" - {metric_name}" not in column:
            continue
        columns.append(column)
    return columns


def read_wandb_wide_csv(path, metric_name):
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if "Step" not in fieldnames:
            raise ValueError(f"CSV file does not contain a `Step` column: {path}")

        columns = metric_columns(fieldnames, metric_name)
        if not columns:
            raise ValueError(f"No `{metric_name}` metric columns found in {path}")

        series = {}
        column_to_label = {}
        for column in columns:
            label = unique_label(algorithm_from_column(column, metric_name), series)
            series[label] = {"steps": [], "values": []}
            column_to_label[column] = label

        for row in reader:
            step_text = row.get("Step", "").strip()
            if not step_text:
                continue
            step = float(step_text)
            for column, label in column_to_label.items():
                value_text = row.get(column, "").strip()
                if not value_text:
                    continue
                series[label]["steps"].append(step)
                series[label]["values"].append(float(value_text))

    return {label: (data["steps"], data["values"]) for label, data in series.items() if data["values"]}


def smooth_values(values, window):
    if window <= 1 or len(values) <= 1:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def ordered_labels(*series_groups):
    labels = set()
    for series in series_groups:
        labels.update(series.keys())
    labels -= EXCLUDED_ALGORITHMS

    ordered = [label for label in ALGORITHM_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def plot_metric(ax, series, labels, title, y_label, x_max, y_bounds=None):
    for label in labels:
        if label not in series:
            continue

        steps, values = series[label]
        line_width = 1.5 if label == "LPG-DRL" else 1.1
        ax.plot(
            steps,
            values,
            label=label,
            color=COLORS.get(label),
            linewidth=line_width,
            alpha=0.95,
        )

    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Episode", labelpad=1)
    ax.set_ylabel(y_label, labelpad=1)
    ax.set_xlim(0, x_max)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)


def prepared_series(series, labels, smooth_window, max_step):
    prepared = {}
    for label in labels:
        if label not in series:
            continue

        steps, values = series[label]
        filtered = [(step, value) for step, value in zip(steps, values) if step <= max_step]
        if not filtered:
            continue

        steps = [step for step, _ in filtered]
        values = [value for _, value in filtered]
        prepared[label] = (steps, smooth_values(values, smooth_window))
    return prepared


def metric_bounds(series):
    values = []
    for _, metric_values in series.values():
        values.extend(metric_values)
    if not values:
        return 0.0, 1.0

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0

    margin = (max_value - min_value) * 0.08
    return min_value - margin, max_value + margin


def scale_points(steps, values, x_min, x_max, y_min, y_max, box):
    left, top, width, height = box
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    points = []
    for step, value in zip(steps, values):
        x = left + (step - x_min) / x_span * width
        y = top + height - (value - y_min) / y_span * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def pdf_number(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def pdf_text(x, y, text, size=7, anchor="start", bold=False, angle=0):
    approx_width = len(str(text)) * size * 0.5
    if angle == 90:
        if anchor == "middle":
            y -= approx_width / 2
        elif anchor == "end":
            y -= approx_width
    elif angle == -90:
        if anchor == "middle":
            y += approx_width / 2
        elif anchor == "end":
            y += approx_width
    else:
        if anchor == "middle":
            x -= approx_width / 2
        elif anchor == "end":
            x -= approx_width

    font = "F2" if bold else "F1"
    escaped = pdf_escape(text)
    if angle == 90:
        matrix = f"0 1 -1 0 {pdf_number(x)} {pdf_number(y)}"
    elif angle == -90:
        matrix = f"0 -1 1 0 {pdf_number(x)} {pdf_number(y)}"
    else:
        matrix = f"1 0 0 1 {pdf_number(x)} {pdf_number(y)}"
    return f"BT /{font} {pdf_number(size)} Tf {matrix} Tm ({escaped}) Tj ET\n"


def pdf_line(x1, y1, x2, y2, width=0.5, color=(0.0, 0.0, 0.0)):
    r, g, b = color
    return (
        f"{pdf_number(r)} {pdf_number(g)} {pdf_number(b)} RG {pdf_number(width)} w "
        f"{pdf_number(x1)} {pdf_number(y1)} m {pdf_number(x2)} {pdf_number(y2)} l S\n"
    )


def pdf_polyline(points, width=0.6, color=(0.0, 0.0, 0.0)):
    if not points:
        return ""

    r, g, b = color
    commands = [f"{pdf_number(r)} {pdf_number(g)} {pdf_number(b)} RG {pdf_number(width)} w"]
    x0, y0 = points[0]
    commands.append(f"{pdf_number(x0)} {pdf_number(y0)} m")
    for x, y in points[1:]:
        commands.append(f"{pdf_number(x)} {pdf_number(y)} l")
    commands.append("S")
    return "\n".join(commands) + "\n"


def pdf_rect(x, y, width, height):
    return f"0 0 0 RG 0.5 w {pdf_number(x)} {pdf_number(y)} {pdf_number(width)} " f"{pdf_number(height)} re S\n"


def pdf_scaled_points(steps, values, x_min, x_max, y_min, y_max, box):
    left, bottom, width, height = box
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    points = []
    for step, value in zip(steps, values):
        x = left + (step - x_min) / x_span * width
        y = bottom + (value - y_min) / y_span * height
        points.append((x, y))
    return points


def write_pdf(output_path, content, width_pt, height_pt):
    stream = content.encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pdf_number(width_pt)} {pdf_number(height_pt)}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii"))
        chunks.append(obj)
        chunks.append(b"\nendobj\n")

    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f\n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n\n".encode("ascii"))
    chunks.append(
        (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n" f"startxref\n{xref_offset}\n%%EOF\n").encode("ascii")
    )
    output_path.write_bytes(b"".join(chunks))


def svg_text(x, y, text, size=14, anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#222">{html.escape(str(text))}</text>'
    )


def svg_rotated_text(x, y, text, size=9, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-family="Arial, sans-serif" '
        f'text-anchor="{anchor}" fill="#222" transform="rotate(-90 {x} {y})">'
        f"{html.escape(str(text))}</text>"
    )


def svg_axis(svg, title, y_label, series, labels, x_bounds, y_bounds, box):
    left, top, width, height = box
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    if title:
        svg.append(svg_text(left + width / 2, top - 16, title, size=10, anchor="middle", weight="bold"))
    svg.append(svg_rotated_text(left - SVG_AXIS_LABEL_OFFSET, top + height / 2, y_label, size=8))
    svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#fff" stroke="#222"/>')

    for tick in range(6):
        ratio = tick / 5
        x = left + ratio * width
        y = top + height - ratio * height
        x_value = x_min + ratio * (x_max - x_min)
        y_value = y_min + ratio * (y_max - y_min)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" stroke="#e3e3e3"/>')
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" stroke="#e3e3e3"/>')
        svg.append(svg_text(x, top + height + 15, f"{x_value:.0f}", size=7, anchor="middle"))
        svg.append(svg_text(left - 5, y + 3, f"{y_value:.1f}", size=7, anchor="end"))

    for label in labels:
        if label not in series:
            continue
        steps, values = series[label]
        points = scale_points(steps, values, x_min, x_max, y_min, y_max, box)
        width_px = 1.8 if label == "LPG-DRL" else 1.25
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS.get(label, "#333")}" '
            f'stroke-width="{width_px}" stroke-linejoin="round" stroke-linecap="round"/>'
        )


def write_metric_svg(
    output_path,
    series,
    labels,
    title,
    y_label,
    x_max,
    figure_width_in,
    figure_height_in,
    y_bounds=None,
):
    width = int(round(figure_width_in * SVG_UNITS_PER_IN))
    height = int(round(figure_height_in * SVG_UNITS_PER_IN))
    plot_margin = 32
    plot_left = plot_margin
    plot_top = 54
    plot_box = (plot_left, plot_top, width - plot_left - 10, height - plot_top - plot_margin)
    x_bounds = (0.0, x_max)
    y_bounds = y_bounds or metric_bounds(series)

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{figure_width_in:g}in" '
            f'height="{figure_height_in:g}in" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 18, title, size=11, anchor="middle", weight="bold"),
    ]

    legend_y = 38
    legend_font_size = 7.5
    legend_line_width = 16
    legend_text_gap = 5
    legend_item_gap = 16
    legend_item_widths = [
        legend_line_width + legend_text_gap + len(str(label)) * legend_font_size * 0.5 for label in labels
    ]
    legend_width = sum(legend_item_widths) + legend_item_gap * max(0, len(labels) - 1)
    legend_x = max(4, width / 2 - legend_width / 2)
    for label, item_width in zip(labels, legend_item_widths):
        x = legend_x
        y = legend_y
        svg.append(
            f'<line x1="{x:.2f}" y1="{y}" x2="{x + legend_line_width:.2f}" y2="{y}" '
            f'stroke="{COLORS.get(label, "#333")}" stroke-width="2"/>'
        )
        svg.append(svg_text(x + legend_line_width + legend_text_gap, y + 3, label, size=legend_font_size))
        legend_x += item_width + legend_item_gap

    svg_axis(svg, "", y_label, series, labels, x_bounds, y_bounds, plot_box)
    svg.append(svg_text(width / 2, plot_top + plot_box[3] + SVG_AXIS_LABEL_OFFSET, "Episode", size=8, anchor="middle"))
    svg.append("</svg>")

    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_metric_pdf(
    output_path,
    series,
    labels,
    title,
    y_label,
    x_max,
    figure_width_in,
    figure_height_in,
    y_bounds=None,
):
    width = figure_width_in * 72.0
    height = figure_height_in * 72.0
    plot_margin = 22.0
    plot_left = plot_margin
    plot_bottom = plot_margin
    plot_box = (plot_left, plot_bottom, width - plot_left - 8.0, height - plot_bottom - 35.0)
    x_bounds = (0.0, x_max)
    y_bounds = y_bounds or metric_bounds(series)
    x_min, x_max_value = x_bounds
    y_min, y_max = y_bounds
    left, bottom, plot_width, plot_height = plot_box

    content = []
    content.append(pdf_text(width / 2, height - 12, title, size=7.3, anchor="middle", bold=True))

    legend_y = height - 25
    legend_font_size = 5.0
    legend_line_width = 12.0
    legend_text_gap = 2.5
    legend_item_gap = 8.0
    legend_item_widths = [
        legend_line_width + legend_text_gap + len(str(label)) * legend_font_size * 0.5 for label in labels
    ]
    legend_width = sum(legend_item_widths) + legend_item_gap * max(0, len(labels) - 1)
    legend_x = max(2.0, width / 2 - legend_width / 2)
    for label, item_width in zip(labels, legend_item_widths):
        x = legend_x
        y = legend_y
        content.append(
            pdf_line(x, y, x + legend_line_width, y, width=1.4, color=hex_to_rgb(COLORS.get(label, "#333333")))
        )
        content.append(pdf_text(x + legend_line_width + legend_text_gap, y - 2.0, label, size=legend_font_size))
        legend_x += item_width + legend_item_gap

    grid_color = (0.88, 0.88, 0.88)
    for tick in range(6):
        ratio = tick / 5.0
        x = left + ratio * plot_width
        y = bottom + ratio * plot_height
        x_value = x_min + ratio * (x_max_value - x_min)
        y_value = y_min + ratio * (y_max - y_min)
        content.append(pdf_line(x, bottom, x, bottom + plot_height, width=0.25, color=grid_color))
        content.append(pdf_line(left, y, left + plot_width, y, width=0.25, color=grid_color))
        content.append(pdf_text(x, bottom - 8.5, f"{x_value:.0f}", size=5.3, anchor="middle"))
        content.append(pdf_text(left - 3.5, y - 1.8, f"{y_value:.1f}", size=5.3, anchor="end"))

    content.append(pdf_rect(left, bottom, plot_width, plot_height))
    content.append(pdf_text(width / 2, bottom - PDF_AXIS_LABEL_OFFSET, "Episode", size=6.3, anchor="middle"))
    content.append(
        pdf_text(left - PDF_AXIS_LABEL_OFFSET, bottom + plot_height / 2, y_label, size=6.3, anchor="middle", angle=90)
    )

    for label in labels:
        if label not in series:
            continue
        steps, values = series[label]
        points = pdf_scaled_points(steps, values, x_min, x_max_value, y_min, y_max, plot_box)
        line_width = 0.8 if label == "LPG-DRL" else 0.55
        content.append(pdf_polyline(points, width=line_width, color=hex_to_rgb(COLORS.get(label, "#333333"))))

    write_pdf(output_path, "".join(content), width, height)


def write_metric_matplotlib(
    output_path,
    series,
    labels,
    title,
    y_label,
    x_max,
    figure_width_in,
    figure_height_in,
    dpi,
    y_bounds=None,
):
    if plt is None:
        raise RuntimeError("matplotlib is required for this output format. Use .pdf or .svg output paths instead.")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(figure_width_in, figure_height_in))
    plot_metric(
        ax,
        series,
        labels,
        title,
        y_label,
        x_max,
        y_bounds=y_bounds,
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(legend_labels)),
        frameon=False,
        fontsize=6,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.01, 0.02, 1, 0.86))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def main():
    args = parse_args()
    reward_series = read_wandb_wide_csv(args.reward_csv, "Reward")
    success_series = read_wandb_wide_csv(args.success_csv, "Success Rate")
    raw_labels = ordered_labels(reward_series, success_series)
    reward_series = prepared_series(reward_series, raw_labels, args.smooth_window, args.max_step)
    success_series = prepared_series(success_series, raw_labels, args.smooth_window, args.max_step)
    labels = ordered_labels(reward_series, success_series)

    if not labels:
        raise RuntimeError(f"No curves with Step <= {args.max_step:g} were found in the provided CSV files.")

    args.reward_output.parent.mkdir(parents=True, exist_ok=True)
    args.success_output.parent.mkdir(parents=True, exist_ok=True)
    if args.smooth_window > 1:
        reward_title = f"Training Reward ({args.smooth_window}-Episode Moving Average)"
        success_title = f"Training Success Rate ({args.smooth_window}-Episode Moving Average)"
    else:
        reward_title = "Training Reward"
        success_title = "Training Success Rate"

    if args.reward_output.suffix.lower() == ".svg":
        write_metric_svg(
            args.reward_output,
            reward_series,
            labels,
            reward_title,
            "Episode Reward",
            args.max_step,
            args.figure_width_in,
            args.figure_height_in,
        )
    elif args.reward_output.suffix.lower() == ".pdf":
        write_metric_pdf(
            args.reward_output,
            reward_series,
            labels,
            reward_title,
            "Episode Reward",
            args.max_step,
            args.figure_width_in,
            args.figure_height_in,
        )
    else:
        write_metric_matplotlib(
            args.reward_output,
            reward_series,
            labels,
            reward_title,
            "Episode Reward",
            args.max_step,
            args.figure_width_in,
            args.figure_height_in,
            args.dpi,
        )

    if args.success_output.suffix.lower() == ".svg":
        write_metric_svg(
            args.success_output,
            success_series,
            labels,
            success_title,
            "Success Rate",
            args.max_step,
            args.figure_width_in,
            args.figure_height_in,
            y_bounds=SUCCESS_RATE_BOUNDS,
        )
    elif args.success_output.suffix.lower() == ".pdf":
        write_metric_pdf(
            args.success_output,
            success_series,
            labels,
            success_title,
            "Success Rate",
            args.max_step,
            args.figure_width_in,
            args.figure_height_in,
            y_bounds=SUCCESS_RATE_BOUNDS,
        )
    else:
        write_metric_matplotlib(
            args.success_output,
            success_series,
            labels,
            success_title,
            "Success Rate",
            args.max_step,
            args.figure_width_in,
            args.figure_height_in,
            args.dpi,
            y_bounds=SUCCESS_RATE_BOUNDS,
        )

    print(f"Saved reward curve to: {args.reward_output}")
    print(f"Saved success-rate curve to: {args.success_output}")
    print(f"Discarded rows with Step > {args.max_step:g}")
    print(f"Moving-average window: {args.smooth_window} episodes")
    print(f"Figure size: {args.figure_width_in:g} in x {args.figure_height_in:g} in")
    print(f"Algorithms plotted: {', '.join(labels)}")


if __name__ == "__main__":
    main()
