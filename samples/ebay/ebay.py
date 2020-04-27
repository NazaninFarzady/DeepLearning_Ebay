import os
import sys
import numpy as np
import datetime
import skimage.draw
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_WEIGHTS_PATH):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)

############################################################
#  Configurations
############################################################

class ObjectConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + object

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################
class ObjectDataset(utils.Dataset):

    def load_object(self, dataset_dir, subset, class_ids=None, ):
        """Load a subset of the COCO dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load image json file
        annFile = os.path.join(dataset_dir, "coco_instances.json")
        coco = COCO(annFile)

        image_dir = os.path.join(dataset_dir, "images")
        # print("image_dir:", image_dir)
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("object", i, coco.loadCats(i)[0]["name"])
        print("class_ids", class_ids)
        # Add images
        for i in image_ids:
            self.add_image(
                "object", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        # print('image_info',image_info)
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # print('annotations', annotations)
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            # print("class_id", class_id)
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    ############################################################
    #  Detection
    ############################################################

    def detect(model, dataset_dir):
        """Run detection on images in the given directory."""
        print("Running on {}".format(dataset_dir))

        # Load image and run detection
        image = skimage.io.imread(dataset_dir)
        # Detect objects
        results = model.detect([image], verbose=1)
        r = results[0]
        # Visualize results
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'])


    ############################################################
    #  Training
    ############################################################

    if __name__ == '__main__':
        import argparse

        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Train Mask R-CNN for Ebay.')
        parser.add_argument("command",
                            metavar="<command>",
                            help="'train' or 'detect' on COCO_ebay")
        parser.add_argument('--dataset', required=True,
                            metavar="/path/to/coco/",
                            help='Directory of the ebay dataset')
        parser.add_argument('--model', required=True,
                            metavar="/path/to/weights.h5",
                            help="Path to weights .h5 file or 'coco'")
        parser.add_argument('--logs', required=False,
                            default=DEFAULT_LOGS_DIR,
                            metavar="/path/to/logs/",
                            help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('--image', required=False,
                            metavar="path or URL to image",
                            help='Image to apply the color splash effect on')
        args = parser.parse_args()
        print("Command: ", args.command)
        print("Model: ", args.model)
        print("Dataset: ", args.dataset)
        print("Logs: ", args.logs)

        # Configurations
        if args.command == "train":
            config = ObjectConfig()
        else:
            class InferenceConfig(ObjectConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1

            inference_config = InferenceConfig()
        inference_config.display()

        # Create model
        if args.command == "train":
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
        else:
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=args.logs)

        # Select weights file to load
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = model.get_imagenet_weights()
        else:
            model_path = args.model

        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

        # Train or evaluate
        if args.command == "train":
            # Training dataset. Use the training set and 35K from the
            # validation set, as as in the Mask RCNN paper.
            dataset_train = ObjectDataset()
            dataset_train.load_object(args.dataset, "train")
            dataset_train.prepare()

            # Validation dataset
            dataset_val = ObjectDataset()
            dataset_val.load_object(args.dataset, "val")
            dataset_val.prepare()

            print("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=30,
                        layers='heads')

        elif args.command == "detect":
            detect(model, args.dataset)
        else:
            print("'{}' is not recognized. "
                  "Use 'train' or 'detect'".format(args.command))
