from torchvision.transforms import ToPILImage
back = ToPILImage()

IMG_DIR = "target_images"
IMG_MASKS_DIR = "target_masks"
CONTEXT_IMG_DIR = "context_images"
CONTEXT_MASKS_DIR = "context_masks"

CONTEXT = 'image_context'
IMAGE = 'image_target'
CONTEXT_MASK = 'mask_context'
IMAGE_MASK = 'mask_target'
OG_SHAPE = "og_shape"
UNLABELED = "unlabeled"
UNLABELED_CONTEXT = "context_unlabeled"
SENTINEL_2 = "sentinel_2"

IDX_REFERENCE = "idx_reference_patch"

STATIC_ZOOM_FACTOR = 7

STONE_ID = 0
NA_AREA_ID = 1
GLACIER_ID = 2
OCEAN_ICE_ID = 3
ID_TO_NAMES = ["Stone", "Na_area","Glacier","Ocean_ice"]

BOUNDING_BOX_DIR = "bounding_boxes"
FRONTS_DIR = "fronts"
SAR_IMAGES_DIR = "sar_images"
ZONES_DIR = "zones"

PARENT_DIR_KEY = "parent_dir"
FRONT_DIR_KEY = "fronts"