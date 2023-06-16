import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def generate_color_palette(image_path, num_colors):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image into a 2D array
    pixels = image.reshape(-1, 3)

    # Perform K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the RGB values of the dominant colors
    colors = kmeans.cluster_centers_

    # Convert the RGB values to integers
    colors = colors.round().astype(int)

    # Generate color names (optional)
    color_names = generate_color_names(colors)

    return colors, color_names


def generate_color_names(colors):
    # Implement color naming logic here (e.g., using color naming libraries or your own scheme)
    # Return a list of color names corresponding to the RGB values

    # Example implementation using random color names
    color_names = []
    for _ in colors:
        color_names.append(get_random_color_name())
    return color_names


def get_random_color_name():
    # Example function to generate random color names
    # Replace with your own color naming logic
    color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Orange']
    return np.random.choice(color_names)


def display_color_palette(colors, color_names):
    # Create a bar graph to display the color palette
    x = np.arange(len(colors))
    plt.bar(x, height=100, width=1, color=[tuple(color / 255) for color in colors])
    plt.xticks(x, color_names, rotation=45)
    plt.xlabel('Color')
    plt.ylabel('Percentage')
    plt.title('Color Palette')
    plt.tight_layout()
    plt.show()


# User input
image_path = input("Enter the path to the image file: ")
num_colors = int(input("Enter the number of colors in the palette: "))

# Generate color palette
colors, color_names = generate_color_palette(image_path, num_colors)

# Display color palette
display_color_palette(colors, color_names)
