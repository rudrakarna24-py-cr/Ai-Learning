import numpy as np

# Set random seed fo reproducibility
np.random.seed(42)

# --- Helper Functions ---
def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# --- 1. Convolution Layer (3x3 Filters) ----
class Conv3x3:
  def __init__(self, num_filters):
    self.num_filters = num_filters
    # Initialize filters using Xavier/He-like small random numbers divided by 9
    self.filters = np.random.randn(num_filters, 3, 3) / (3 * 3)

  def forward(self, input_image):
    # input_image shape: (H, W)
    self.last_input = input_image
    h, w = input_image.shape
                 
