import numpy as np
import cv2

def calculate_centroid(edge_points):
    return np.mean(edge_points, axis=0)

def calculate_covariance(edge_points, centroid):
    shifted_points = edge_points - centroid
    return np.dot(shifted_points.T, shifted_points) / shifted_points.shape[0]

def find_major_axis(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    return major_axis

def calculate_rotation_angle(major_axis):
    return np.arctan2(major_axis[1], major_axis[0])

def rotate_image(image, angle, centroid):
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    height, width = image.shape
    rotated_image = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            x_shifted, y_shifted = x - centroid[0], y - centroid[1]
            x_rotated = cos_angle * x_shifted - sin_angle * y_shifted + centroid[0]
            y_rotated = sin_angle * x_shifted + cos_angle * y_shifted + centroid[1]
            if 0 <= x_rotated < width and 0 <= y_rotated < height:
                rotated_image[int(y_rotated), int(x_rotated)] = image[y, x]
    
    return rotated_image

# Example edge points
edge_points = np.array([[x, y] for x in range(100) for y in range(100) if (x-50)**2 + (y-50)**2 < 400])

# Calculate centroid and covariance
centroid = calculate_centroid(edge_points)
covariance_matrix = calculate_covariance(edge_points, centroid)

# Find the major axis
major_axis = find_major_axis(covariance_matrix)

# Calculate the rotation angle needed
rotation_angle = calculate_rotation_angle(major_axis)

# Assuming you have a grayscale image array `image`
# Rotate the image
corrected_image = rotate_image(image, -rotation_angle, centroid)

cv2.imshow('Input', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
