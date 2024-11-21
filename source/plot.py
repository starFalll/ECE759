import matplotlib.pyplot as plt
import os

# Function to plot the time cost and save the result to a PDF
def plot_time_cost(file_name, output_pdf):
    # Read the time costs from the file
    with open(file_name, 'r') as file:
        times = [float(line.strip()) for line in file.readlines()]

    # Number of threads 
    threads = list(range(1, len(times) + 1))

    # Create the plot
    plt.plot(threads, times, marker='o', label=file_name)
    schedule_type, chunk, _, max_thread, _ = file_name.split('_')
    schedule_type = schedule_type.split('/')[-1]
    # Add labels and title
    plt.xlabel('Number of Threads')
    plt.ylabel('Time Cost (ms)')
    plt.title('Time Cost vs. Threads - {}={}, max threads={}'.format(schedule_type, chunk, max_thread))
    plt.grid(True)
    plt.legend()

    # Save the plot as a PDF
    plt.savefig(output_pdf)

    # Optionally, show the plot
    # plt.show()

    # Close the plot to avoid overlap in subsequent plots
    plt.clf()

file_names = []
# Example usage with different files
for root, _, files in os.walk("../results/"):
    for file in files:
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
print(file_names)
for name in file_names:
    plot_time_cost('../results/{}.txt'.format(name), '../results/{}.pdf'.format(name))