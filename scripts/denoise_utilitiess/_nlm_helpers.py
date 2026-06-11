import numpy as np
from numba import jit


# Function to preprocess neighbors (small_window x small_window) for each pixel
@jit(nopython=True)
def findAllNeighbors(padImg, small_window, big_window, h, w):
    """
    Preprocess neighbors (small_window x small_window) for each pixel.
    """
    smallWidth = small_window // 2
    bigWidth = big_window // 2
    neighbors = np.zeros((padImg.shape[0], padImg.shape[1], small_window, small_window))
    for i in range(bigWidth, bigWidth + h):
        for j in range(bigWidth, bigWidth + w):
            neighbors[i, j] = padImg[(i - smallWidth):(i + smallWidth + 1), (j - smallWidth):(j + smallWidth + 1)]
    return neighbors


# Function to calculate the weighted average value (Ip) for each pixel
@jit(nopython=True)
def evaluateNorm(pixelWindow, neighborWindow, Nw):
    """
    Calculate the weighted average value (Ip) for each pixel.
    """
    Ip_Numerator, Z = 0.0, 0.0
    for i in range(neighborWindow.shape[0]):
        for j in range(neighborWindow.shape[1]):
            q_window = neighborWindow[i, j]
            q_x, q_y = q_window.shape[0] // 2, q_window.shape[1] // 2
            Iq = q_window[q_x, q_y]
            w = np.exp(-1 * ((np.sum((pixelWindow - q_window) ** 2)) / Nw))
            Ip_Numerator += w * Iq
            Z += w
    return Ip_Numerator / Z if Z > 0 else 0.0


# Template for a custom Non-Local Means implementation (currently commented out)
# class NLMeansCustom:
#     """
#     Custom Non-Local Means implementation for grayscale images.
#     """
#     def apply(self, img, h=30, small_window=7, big_window=21):
#         """
#         Apply NLM filtering to a 2D grayscale image.
#         Parameters:
#             img (np.ndarray): Grayscale image.
#             h (float): Filtering parameter (higher removes more noise).
#             small_window (int): Patch size.
#             big_window (int): Search window size.
#         Returns:
#             np.ndarray: Denoised image (float64).
#         """
#         # Padding the original image with reflect mode
#         padImg = np.pad(img, big_window // 2, mode='reflect')
#         return self.NLM(padImg, img, h, small_window, big_window)
#
#     @staticmethod
#     def NLM(padImg, img, h, small_window, big_window):
#         """
#         Core NLM algorithm for denoising.
#         """
#         Nw = (h ** 2) * (small_window ** 2)
#         h_img, w_img = img.shape
#         result = np.zeros(img.shape)
#         bigWidth = big_window // 2
#         smallWidth = small_window // 2
#         neighbors = findAllNeighbors(padImg, small_window, big_window, h_img, w_img)
#         for i in range(bigWidth, bigWidth + h_img):
#             for j in range(bigWidth, bigWidth + w_img):
#                 pixelWindow = neighbors[i, j]
#                 neighborWindow = neighbors[(i - bigWidth):(i + bigWidth + 1), (j - bigWidth):(j + bigWidth + 1)]
#                 Ip = evaluateNorm(pixelWindow, neighborWindow, Nw)
#                 result[i - bigWidth, j - bigWidth] = max(min(255, Ip), 0)
#         return result
