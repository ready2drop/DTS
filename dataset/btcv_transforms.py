from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CenterSpatialCropd,
    CropForegroundd,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    ToTensord,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)

class Transforms:
    def __init__(
        self,
        data_name: str,
        spatial_size: int = 96,
        image_size: int = 96,
        num_samples: int = 1,
        label_smoothing: str = "k-nls",
    ):
        self.data_name = data_name
        self.spatial_size = spatial_size
        self.image_size = image_size
        self.num_samples = num_samples
        self.label_smoothing = label_smoothing

    def generate(self):
        transform = {}
        transform["train"] = [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(self.spatial_size, self.image_size, self.image_size),
                pos=1,
                neg=1,
                num_samples=self.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            # CenterSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=[self.spatial_size, self.image_size, self.image_size]
            # ),

            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),

            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]

        transform["val"] = [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            # Resized(keys=["image", "label"], spatial_size=(spatial_size, image_size, image_size)),
            ToTensord(keys=["image", "label"]),
        ]

        transform["test"] = [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            ToTensord(keys=["image"]),
        ]

        for k, v in transform.items():
            if self.label_smoothing and k == "train": v.pop(0)
            transform[k] = Compose(v)

        return transform