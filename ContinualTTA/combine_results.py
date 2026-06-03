import os
import pandas as pd

ALL_METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]
ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

def combine_csvs():
    combined_data = {"corruption": ALL_CORRUPTIONS + ["Mean"]}
    available_methods = []

    for method in ALL_METHODS:
        filepath = f"results/imagenetc_{method}.csv"
        if os.path.exists(filepath):
            available_methods.append(method)
            df = pd.read_csv(filepath)
            # Map the accuracies to our combined dict
            combined_data[method] = df[method].tolist()
        else:
            print(f"Warning: {filepath} not found. Skipping {method}.")

    if not available_methods:
        print("No CSV files found in 'results/'. Run the python scripts first!")
        return

    # Create final dataframe and save it
    final_df = pd.DataFrame(combined_data)
    final_df.to_csv("results/imagenetc_ALL_COMBINED.csv", index=False)
    print("Combined CSV saved to: results/imagenetc_ALL_COMBINED.csv\n")

    generate_latex(final_df, available_methods)

def generate_latex(df, methods):
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{ContinualTTA (Ours)}",
    }
    corr_names = {
        "gaussian_noise": "Gaussian Noise", "shot_noise": "Shot Noise",
        "impulse_noise": "Impulse Noise",   "defocus_blur": "Defocus Blur",
        "glass_blur": "Glass Blur",         "motion_blur": "Motion Blur",
        "zoom_blur": "Zoom Blur",           "snow": "Snow",
        "frost": "Frost",                   "fog": "Fog",
        "brightness": "Brightness",         "contrast": "Contrast",
        "elastic_transform": "Elastic",     "pixelate": "Pixelate",
        "jpeg_compression": "JPEG",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Accuracy (\%) on ImageNet-C under continual "
                 r"sequential shift, Severity~5. "
                 r"\textbf{Bold} = best per row. "
                 r"Source model: ResNet-50 pretrained on clean ImageNet.}")
    lines.append(r"\label{tab:main_imagenetc}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append(r"\toprule")
    lines.append("Corruption & " + " & ".join(cite[m] for m in methods) + r" \\")
    lines.append(r"\midrule")

    for i in range(len(ALL_CORRUPTIONS)):
        row_data = df.iloc[i]
        corruption = row_data["corruption"]
        best = max(row_data[m] for m in methods)
        
        row_str = corr_names[corruption]
        for m in methods:
            val = row_data[m]
            if abs(val - best) < 0.05:
                row_str += f" & \\textbf{{{val:.1f}}}"
            else:
                row_str += f" & {val:.1f}"
        lines.append(row_str + r" \\")

    lines.append(r"\midrule")
    
    # Mean row
    mean_data = df.iloc[-1]
    best_m = max(mean_data[m] for m in methods)
    row_m = r"\textbf{Mean}"
    for m in methods:
        val = mean_data[m]
        if abs(val - best_m) < 0.05:
            row_m += f" & \\textbf{{{val:.1f}}}"
        else:
            row_m += f" & {val:.1f}"
    lines.append(row_m + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)
    with open("results/imagenetc_table.tex", "w") as f:
        f.write(latex_str)
        
    print("="*70)
    print("LaTeX Table Generated (Saved to: results/imagenetc_table.tex)")
    print("="*70)
    print(latex_str)

if __name__ == "__main__":
    combine_csvs()