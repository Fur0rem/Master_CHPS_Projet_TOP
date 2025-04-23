"""
@file scripts/layouts_strong_scaling.py
@brief Script to run the strong scaling benchmark for the layouts of A, B, and C.
"""

# For running the benchmark
import subprocess
import os

# For plotting the results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


MAX_THREADS = int(subprocess.check_output("lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l", shell=True).decode("utf-8").strip())
print("Max threads:", MAX_THREADS)

# Build the benchmark executable
subprocess.run(["cmake", "--build", "build"])

def launch_with_nb_threads(executable: str, nb_threads: int) -> str:
	"""
	Launch the benchmark with the given number of threads and return the output.
	"""
	# Set the environment variables for OpenMP
	env = os.environ.copy()
	env["OMP_PROC_BIND"] = "true"
	env["OMP_PLACES"] = "cores"
	env["OMP_NUM_THREADS"] = str(nb_threads)

	# Launch the benchmark
	result = subprocess.Popen(
		[f"./build/benchmarks/{executable}", f"--kokkos-num-threads={nb_threads}"],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		env=env
	)
	stdout, stderr = result.communicate()
	
	# Check for errors
	stderr = stderr.decode("utf-8")
	if stderr != "":
		print("Error:", stderr)

	print(stdout.decode("utf-8"))
	
	# Return the output
	return stdout.decode("utf-8")


def parse_output(output: str) -> dict:
	"""
	Parse the output of the benchmark and return a dictionary with the results.
	"""
	# Output format:
	# Name : [Min: Xs, Max: Ys, Med: Zs]

	results = {}
	for line in output.split("\n"):
		if line.startswith("Ar_") or line.startswith("Al_"):
			values = line.split(",")
			name = values[0]
			min = max = med = None
			for value in values:
				if "Min" in value:
					min = float(value.split(":")[1].strip()[:-1])
				elif "Max" in value:
					max = float(value.split(":")[1].strip()[:-1])
				elif "Med" in value:
					med = float(value.split(":")[1].strip()[:-1])
			results[name] = {
				"min": min,
				"max": max,
				"med": med
			}
	return results


def run_benchmark(executable: str, result_name: str):
	"""
	Run the benchmark with the given executable and plot the results.
	"""

	all_results = {}
	for n_threads in range(1, MAX_THREADS + 1):
		print(f"Running with {n_threads} threads")
		all_results[n_threads] = parse_output(launch_with_nb_threads(executable, n_threads))

	print(all_results)

	# Re-arrange the results to have an array {name: [(min, max, med), ... for each n_threads]}
	results = {}
	for n_threads in all_results:
		for name in all_results[n_threads]:
			if name not in results:
				results[name] = []
			results[name].append((all_results[n_threads][name]["min"], all_results[n_threads][name]["max"], all_results[n_threads][name]["med"]))

	# Create a figure with GridSpec for two subplots
	fig = plt.figure(figsize=(14, 6))
	gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio for plot and table


	markers = {
		"Ar_Br_Cr": ("#000000", "o", "none"),
		"Ar_Br_Cl": ("#555555", "o", "full"),
		"Ar_Bl_Cr": ("#000088", "s", "none"),
		"Ar_Bl_Cl": ("#0000FF", "s", "full"),
		"Al_Br_Cr": ("#008800", "^", "none"),
		"Al_Br_Cl": ("#00FF00", "^", "full"),
		"Al_Bl_Cr": ("#880000", "*", "none"),
		"Al_Bl_Cl": ("#FF0000", "*", "full"),
	}

	# Plot the data with error bars in the first subplot
	ax_plot = fig.add_subplot(gs[0])
	for name in results:
		if name in markers:
			x = list(range(1, MAX_THREADS + 1))  # Number of threads
			y_med = [result[2] for result in results[name]]  # Median values
			y_err = [(result[1] - result[0]) / 2 for result in results[name]]  # Error bars (half the range)
			colour, marker, fillstyle = markers[name]
			ax_plot.errorbar(x, y_med, yerr=y_err, label=name, 
							marker=marker, markersize=10, fillstyle=fillstyle,
							color=colour, linestyle="dotted", linewidth=2)
		else:
			print(f"Warning: {name} not in colours dictionary")

	# Plot the times
	ax_plot.set_xlabel("Number of threads")
	ax_plot.set_ylabel("Runtime (s)")
	ax_plot.grid()

	# Change font size for the plot
	for label in (ax_plot.get_xticklabels() + ax_plot.get_yticklabels()):
		label.set_fontsize(13)
	# Change font size of the axes titles
	ax_plot.xaxis.label.set_size(16)
	ax_plot.yaxis.label.set_size(16)


	# Add the table in the second subplot
	ax_table = fig.add_subplot(gs[1])
	ax_table.axis("off")  # Turn off the axis for the table

	# Prepare the table data
	columns = ["Marker", "Layout of A", "Layout of B", "Layout of C"]
	rows = []
	colours_to_set = []
	for name in results:
		row = []
		# Add the appropriate symbol for the colour
		symbol_colour, symbol_name, symbol_fillstyle = markers[name]
		if symbol_name == "o":
			row.append("●" if symbol_fillstyle == "full" else "○")
			colours_to_set.append((len(rows) + 1, len(row) - 1, symbol_colour))
		elif symbol_name == "s":
			row.append("■" if symbol_fillstyle == "full" else "□")
			colours_to_set.append((len(rows) + 1, len(row) - 1, symbol_colour))
		elif symbol_name == "^":
			row.append("▲" if symbol_fillstyle == "full" else "△")
			colours_to_set.append((len(rows) + 1, len(row) - 1, symbol_colour))
		elif symbol_name == "*":
			row.append("★" if symbol_fillstyle == "full" else "☆")
			colours_to_set.append((len(rows) + 1, len(row) - 1, symbol_colour))
		else:
			print(f"Warning: {symbol_name} not in markers dictionary")
			row.append(" ")
			
		# Determine the layout of A, B, and C
		row.append("Right" if "Ar" in name else "Left")
		row.append("Right" if "Br" in name else "Left")
		row.append("Right" if "Cr" in name else "Left")
		rows.append(row)

	# Create the table
	table = ax_table.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
	
	# Adjust table font size and scale
	table.auto_set_font_size(False)
	table.set_fontsize(13)
	table.scale(1.7, 3.08)

	# Set the table cell colours
	for i, j, colour in colours_to_set:
		cell = table[(i, j)]
		cell.set_text_props(color=colour)

	# Save the plot and write the results to a file
	plt.tight_layout()
	plt.savefig(f"results/{result_name}.svg")
	plt.savefig(f"results/{result_name}.png")
	with open(f"results/{result_name}.log", "w") as f:
		for n_threads in all_results:
			f.write(f"Threads: {n_threads}\n")
			for name in all_results[n_threads]:
				f.write(f"{name}: {all_results[n_threads][name]}\n")
			f.write("\n")

run_benchmark("top.benchmark_layout_all", "strong_scaling_layout_all")
run_benchmark("top.benchmark_layout_minus_outliers", "strong_scaling_layout_minus_outliers")