class ImagePair:
    def __init__(self, I_s, I_t, L_s, L_t):
        # Source image
        self.I_s = I_s

        # Target image
        self.I_t = I_t

        # Source lighting
        self.L_s = L_s

        # Target lighting
        self.L_t = L_t

# Loads all the data from the dataset at "path" into a Python list
# of ImagePairs
def load_data(path):
    return None