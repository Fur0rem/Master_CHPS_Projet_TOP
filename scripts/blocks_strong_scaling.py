"""
@file scripts/layouts_strong_scaling.py
@brief Script to run the strong scaling benchmark for the layouts of A, B, and C.
"""

# For running the benchmark
import subprocess
import os

import gc


# For plotting the results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


MAX_THREADS = int(subprocess.check_output("lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l", shell=True).decode("utf-8").strip()) * 2
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
	gc.disable()
	result = subprocess.run(
		[f"./build/benchmarks/{executable}", f"--kokkos-num-threads={nb_threads}"],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		env=env
	)
	gc.enable()
	stdout, stderr = result.stdout, result.stderr
	
	# Check for errors
	stderr = stderr.decode("utf-8")
	if stderr != "":
		print("Error:", stderr)

	print(stdout.decode("utf-8"))
	
	# Return the output
	return stdout.decode("utf-8")

def run_benchmark(executable: str) -> dict:
	"""
	Run a benchmark with the number of threads going from 1 to MAX_THREADS.
	Returns an array of all the outputs of all programs
	"""
	all_results = {}
	for n_threads in range(1, MAX_THREADS + 1):
		print(f"Running with {n_threads} threads")
		all_results[str(n_threads)] = launch_with_nb_threads(executable, n_threads)
	
	return all_results

def parse_output(output: str) -> dict:
	"""
	Parse the output of the benchmark and return a dictionary with the results.
	"""
	# Output format:
	# Name : [Min: Xs, Max: Ys, Med: Zs]

	results = {}
	for line in output.split("\n"):
		if line == "":
			continue
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

outputs = run_benchmark("top.benchmark_cache_blocking")

for key, value in outputs.items():
	outputs[key] = parse_output(value)

# Separate the i and ij results
i_results = {}
ij_results = {}
non_cache_blocked_results = {}
for key, value in outputs.items():
	for name, result in value.items():
		if "i" in name and "Cache Blocked" in name and not "ij" in name:
			# Rename it to block size I
			name = name.split(" ")[-1]
			name = name.split(",")[0]
			name = name.replace("Cache Blocked ", "")
			name = name.replace("i", "")
			name = "Block size " + name
			i_results[key] = i_results.get(key, {})
			i_results[key][name] = result
		elif "Cache Blocked ij" in name:
			# Rename it to block size IxJ
			name = name.split(" ")[-1]
			name = name.split(",")[0]
			name = name.replace("Cache Blocked ", "")
			name = name.replace("ij", "")
			name = "Block size " + name + "x" + name
			ij_results[key] = ij_results.get(key, {})
			ij_results[key][name] = result
		elif "No Cache Blocking" in name:
			non_cache_blocked_results[key] = non_cache_blocked_results.get(key, {})
			non_cache_blocked_results[key][name] = result
		else:
			print("Unknown name:", name)
			continue

# Get max time of every single computation
max_time = -1
for key, value in outputs.items():
	for name, result in value.items():
		if "Cache Blocked" in name:
			max_time = max(max_time, result["max"])
		elif "No Cache Blocking" in name:
			max_time = max(max_time, result["max"])

def plot_results(results: str, max_time: float, where_to_save: str) -> None:
	"""
	Plot the results of the benchmark.
	"""
	# Plot the strong scaling of i (with no cache blocking too)
	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)
	ax.set_xlabel("Number of threads")
	ax.set_ylabel("Runtime (s)")

	# Set the y-lim to the max time to be the same for all plots
	ax.set_ylim(0, max_time * 1.1)
	ax.set_xlim(0, MAX_THREADS + 1)

	markers = {
		"4": ("#880000", "o", "none"),
		"4x4": ("#880000", "o", "full"),
		
		"8": ("#FF0000", "o", "full"),
		"8x8": ("#FF0000", "o", "none"),
		
		"16": ("#000088", "s", "none"),
		"16x16": ("#000088", "s", "full"),
		
		"32": ("#0000FF", "s", "full"),
		"32x32": ("#0000FF", "s", "none"),
		
		"64": ("#008800", "^", "none"),
		"64x64": ("#008800", "^", "full"),
		
		"128": ("#00FF00", "^", "full"),
		"128x128": ("#00FF00", "^", "none"),
	}

	x = [i for i in range(1, MAX_THREADS + 1)]
	
	if results == "i":
		results_dict = i_results
	elif results == "ij":
		results_dict = ij_results
	else:
		print("Unknown results:", results)
	
	for name, _ in results_dict["1"].items():
		block_size = name.split(" ")[-1]
		
		y_med = []
		y_min = []
		y_max = []
		for key in results_dict.keys():
			y_med.append(results_dict[key][name]["med"])
			y_min.append(results_dict[key][name]["min"])
			y_max.append(results_dict[key][name]["max"])
		y_err = [(y_max[i] - y_med[i]) / 2 for i in range(len(y_med))] # Error bars (half the range)
		
		ax.errorbar(x, y_med, yerr=y_err, label=name, linestyle="dotted", color=markers[block_size][0], marker=markers[block_size][1], markerfacecolor=markers[block_size][0])
	
	# Add the non cache blocked results
	y_med = []
	y_min = []
	y_max = []
	for key in non_cache_blocked_results.keys():
		y_med.append(non_cache_blocked_results[key]["No Cache Blocking"]["med"])
		y_min.append(non_cache_blocked_results[key]["No Cache Blocking"]["min"])
		y_max.append(non_cache_blocked_results[key]["No Cache Blocking"]["max"])
	y_err = [(y_max[i] - y_med[i]) / 2 for i in range(len(y_med))] # Error bars (half the range)
	
	ax.errorbar(x, y_med, yerr=y_err, label="No Cache Blocking", color="black", marker="*", markerfacecolor="black")

	# Change font size for the plot
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(13)
	# Change font size of the axes titles
	ax.xaxis.label.set_size(16)
	ax.yaxis.label.set_size(16)

	plt.legend()
	plt.grid()
	
	plt.savefig(f"results/strong_scaling_{where_to_save}.png", bbox_inches='tight')
	plt.savefig(f"results/strong_scaling_{where_to_save}.svg", bbox_inches='tight')
	with open(f"results/strong_scaling_{where_to_save}.log", "w") as f:
		for n_threads in results_dict:
			f.write(f"Threads: {n_threads}\n")
			f.write(f"No Cache Blocking: {non_cache_blocked_results[n_threads]['No Cache Blocking']}\n")
			for name in results_dict[n_threads]:
				f.write(f"{name}: {results_dict[n_threads][name]}\n")
			f.write("\n")


plot_results("i", max_time, "cache_blocking_i")
plot_results("ij", max_time, "cache_blocking_ij")