import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

# Function to create a custom data loader
def custom_data_loader(data_dir, batch_size, image_size, mask_size, data_mode, val_split):

    # Function to load and preprocess the images and masks
    def load_and_preprocess_image(image_path):
        # Load image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, image_size[0], image_size[1])
        img = img / 127.5 -1  # Scale to [-1, 1]
        return img

    def load_and_preprocess_mask(mask_path):
        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.resize_with_pad(mask, mask_size[0], mask_size[1])
        mask = mask // 255  # Convert mask values to 0 or 1
        return mask

    # Function to get image and mask file paths
    def get_image_and_mask_paths(data_dir):
        image_dir = os.path.join(data_dir, f"{data_mode}_images")
        mask_dir = os.path.join(data_dir, f"{data_mode}_masks")

        image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)]
        print(len(image_paths), len(mask_paths))
        return image_paths, mask_paths
    
    # Function to create the dataset from image and mask paths
    def create_dataset(image_paths, mask_paths, batch_size):
        image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        mask_dataset = tf.data.Dataset.from_tensor_slices(mask_paths)

        # Load and preprocess images and masks
        image_dataset = image_dataset.map(load_and_preprocess_image)
        mask_dataset = mask_dataset.map(load_and_preprocess_mask)

        # Combine image and mask datasets
        dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))


        # Shuffle and batch the dataset
        #dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.batch(batch_size)

        return dataset
    
    
    ##########

    image_paths, mask_paths = get_image_and_mask_paths(data_dir)

    # Split the data into train and validation sets
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )
    # Create train and validation datasets
    train_dataset = create_dataset(train_image_paths, train_mask_paths, batch_size)
    val_dataset = create_dataset(val_image_paths, val_mask_paths, batch_size)

    return train_dataset, val_dataset
