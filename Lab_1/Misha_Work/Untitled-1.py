# %%
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
from astropy.io import fits
from scipy import stats, optimize

# %%
def create_master_dark(file_template, num_frames):
    """
    Create a master dark frame by median-combining dark frames.
    
    :param file_template: Template path for the dark frame files with placeholders for the frame number.
    :type file_template: str
    :param num_frames: The number of dark frames to combine.
    :type num_frames: int
    :return: Master dark frame (2D numpy array), Stack of individual dark frames (3D numpy array)
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    # List to store the data from each dark frame
    dark_frames_data = []
    # Loop through each dark frame and read the data
    for i in range(num_frames):
        # Construct the file name based on the frame number
        file_path = file_template.format(i)
        # Open the FITS file and extract the data
        with fits.open(file_path) as hdul:
            dark_frames_data.append(hdul[0].data)
    # Stack the frames into a 3D array (stack of 2D frames)
    stacked_frames = np.stack(dark_frames_data, axis=0)
    # Calculate the median across the stack (axis=0)
    master_dark_frame = np.median(stacked_frames, axis=0)
    
    return master_dark_frame, stacked_frames


# Template for the file names (with placeholder for frame number)
file_template = r"..\FITS_Files\3.2.3_10_dark_frames_30_sec.{:08d}.DARK.FIT"
# Number of dark frames to combine
num_frames = 10
# Create the master dark frame and get the individual frames
master_dark_frame, dark_frames = create_master_dark(file_template, num_frames)

# %%
# Calculate the median value of the master dark frame
median_value = np.median(master_dark_frame)

# Output the median value
print(f"Median value of the master dark frame: {median_value}")

# %%
# Flattening the data into a 1D array
flattened_data = master_dark_frame.flatten()

# Plot the histogram of pixel intensities for the median dark frame
plt.figure()
plt.hist(flattened_data, range = (np.min(flattened_data), np.max(flattened_data)), bins=200, log=True, color='blue', edgecolor='black')
plt.title('Pixel Intensity Distribution of the Master Dark Frame')
plt.xlabel('Pixel Intensity (Counts)')
plt.ylabel('Number of Pixels')
plt.grid(True)

# %%
# Filtering the data to include only values less than or equal to the median maximum
med_max = 1050
filtered_data = flattened_data[flattened_data <= med_max]

# Plot the new histogram 
plt.figure()
plt.hist(filtered_data, range = (np.min(filtered_data), np.max(filtered_data)), bins=200, color='blue', edgecolor='black')
plt.title('Pixel Intensity Distribution of the Master Dark Frame (<= 1050)')
plt.xlabel('Pixel Intensity (Counts)')
plt.ylabel('Number of Pixels')
plt.grid(True)

# %%
# Initialize lists for hot and warm pixels
hot_pixels = []
warm_pixels = []

# Iterate over each pixel in the median dark frame
for i in range(master_dark_frame.shape[0]):
    for j in range(master_dark_frame.shape[1]):
        if master_dark_frame[i, j] >= 20000:  # Hot pixel threshold
            hot_pixels.append([i, j])  # Store as [row, col] (i, j)
        elif 10000 < master_dark_frame[i, j] < 20000:  # Warm pixel range
            warm_pixels.append([i, j])  # Store as [row, col] (i, j)

# Calculate the total number of pixels in the median dark frame
total_pixels = master_dark_frame.size

# Calculate the fraction of rejected pixels
fraction_of_rejected_pixels = (len(hot_pixels) + len(warm_pixels)) / total_pixels

# Output the results
print(f"Hot pixels: {hot_pixels}")
print(f"Warm pixels: {warm_pixels}")

# %%
# Initialize lists for visibility checks
hot_pixels_in_all_frames = []
warm_pixels_in_all_frames = []
hot_pixels_in_median_only = []
warm_pixels_in_median_only = []
hot_pixels_in_both = []
warm_pixels_in_both = []

# Check each pixel in the list of hot and warm pixels
for pixel in hot_pixels:
    i, j = pixel
    appears_in_all_frames = all(dark_frame[i, j] >= 20000 for dark_frame in dark_frames)
    appears_in_median_only = (master_dark_frame[i, j] >= 20000) and not appears_in_all_frames
    
    if appears_in_all_frames:
        hot_pixels_in_all_frames.append(pixel)
    elif appears_in_median_only:
        hot_pixels_in_median_only.append(pixel)
    else:
        hot_pixels_in_both.append(pixel)

for pixel in warm_pixels:
    i, j = pixel
    appears_in_all_frames = all(10000 < dark_frame[i, j] < 20000 for dark_frame in dark_frames)
    appears_in_median_only = (10000 < master_dark_frame[i, j] < 20000) and not appears_in_all_frames
    
    if appears_in_all_frames:
        warm_pixels_in_all_frames.append(pixel)
    elif appears_in_median_only:
        warm_pixels_in_median_only.append(pixel)
    else:
        warm_pixels_in_both.append(pixel)

# Output the results
print(f"Hot pixels in all frames: {hot_pixels_in_all_frames}")
print(f"Warm pixels in all frames: {warm_pixels_in_all_frames}")
print(f"Hot pixels only in median: {hot_pixels_in_median_only}")
print(f"Warm pixels only in median: {warm_pixels_in_median_only}")
print(f"Hot pixels in both median and some frames: {hot_pixels_in_both}")
print(f"Warm pixels in both median and some frames: {warm_pixels_in_both}")

# %%
def combine_dark_frames_with_bias(dark_file_template, exposure_times, bias_file):
    """
    Combine dark frames with varying exposure times with the bias frame by adding their data together.
    
    :param dark_file_template: Template path for the dark frame files with placeholders for the exposure time.
    :type dark_file_template: str
    :param exposure_times: List of exposure times for the dark frames.
    :type exposure_times: list
    :param bias_file: Path to the bias frame FITS file.
    :type bias_file: str
    :return: Stacked combined dark and bias frames (3D numpy array), Stacked individual dark frames (3D numpy array)
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    # Open the bias frame and get the data
    with fits.open(bias_file) as hdul_bias:
        bias_data = hdul_bias[0].data
    # Initialize lists to hold the combined data and individual dark frame data
    combined_data = []
    individual_dark_frames = []
    # Loop over each exposure time and process the corresponding dark frame
    for exposure_time in exposure_times:
        # Construct the file path for the current dark frame
        dark_file_path = dark_file_template.format(exposure_time)
        # Open the dark frame and get the data
        with fits.open(dark_file_path) as hdul_dark:
            dark_data = hdul_dark[0].data
        # Store the individual dark frame data
        individual_dark_frames.append(dark_data)
        # Add the dark frame to the bias frame
        combined_frame = dark_data + bias_data
        # Store the combined frame data
        combined_data.append(combined_frame)
    # Stack the combined frames and individual dark frames into 3D arrays
    stacked_combined_data = np.stack(combined_data, axis=0)
    stacked_individual_dark_frames = np.stack(individual_dark_frames, axis=0)
    
    return stacked_combined_data, stacked_individual_dark_frames

# Example usage
dark_file_template = r"../FITS_Files/3.2.2_dark_frame_{:d}_sec.00000000.DARK.FIT"
exposure_times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]  # List of exposure times for the dark frames
bias_file = r"../FITS_Files/3.1_bias.00000011.BIAS.FIT"

# Combine the dark frames with the bias frame
stacked_combined_data, stacked_dark_frames = combine_dark_frames_with_bias(dark_file_template, exposure_times, bias_file)


