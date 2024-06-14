import math
from enum import Enum
from typing import List, Union

# only 57 types
COLOR_PALETTE = [
    0,
    0,
    0,
    228,
    26,
    28,
    55,
    126,
    184,
    77,
    175,
    74,
    152,
    78,
    163,
    255,
    127,
    0,
    255,
    255,
    51,
    166,
    86,
    40,
    247,
    129,
    191,
    153,
    153,
    153,
    102,
    194,
    165,
    252,
    141,
    98,
    141,
    160,
    203,
    231,
    138,
    195,
    166,
    216,
    84,
    255,
    217,
    47,
    229,
    196,
    148,
    179,
    179,
    179,
    141,
    211,
    199,
    255,
    255,
    179,
    190,
    186,
    218,
    251,
    128,
    114,
    128,
    177,
    211,
    253,
    180,
    98,
    179,
    222,
    105,
    252,
    205,
    229,
    217,
    217,
    217,
    188,
    128,
    189,
    204,
    235,
    197,
    255,
    237,
    111,
    166,
    206,
    227,
    31,
    120,
    180,
    178,
    223,
    138,
    51,
    160,
    44,
    251,
    154,
    153,
    227,
    26,
    28,
    253,
    191,
    111,
    255,
    127,
    0,
    202,
    178,
    214,
    106,
    61,
    154,
    255,
    255,
    153,
    177,
    89,
    40,
    127,
    201,
    127,
    190,
    174,
    212,
    253,
    192,
    134,
    255,
    255,
    153,
    56,
    108,
    176,
    240,
    2,
    127,
    191,
    91,
    22,
    102,
    102,
    102,
    27,
    158,
    119,
    217,
    95,
    2,
    117,
    112,
    179,
    231,
    41,
    138,
    102,
    166,
    30,
    230,
    171,
    2,
    166,
    118,
    29,
    102,
    102,
    102,
]

OPACITY_DARK = 200
OPACITY_THIN = 50
KEYPOINT_MIN_STROKE_WIDTH = 2
POSE_ESTIMATION_MIN_STROKE_WIDTH = 7
SEPARATOER = "@@@@@"

EXPORT_IMAGE_WITH_ANNOTATIONS_SUPPORTED_IMAGE_TYPES = (
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
)

# under 512 MB. Actual size is 536870888 bytes, but to consider other attributes,
# minus 888 bytes.
# Because of V8's limitation, API only can accept the JSON string that length is
# under this.
SUPPORTED_CONTENTS_SIZE = 536870000

# API can accept under 250 MB
SUPPORTED_VIDEO_SIZE = 250 * math.pow(1024, 2)

# API can accept under 20 MB
SUPPORTED_IMAGE_SIZE = 20 * math.pow(1024, 2)

# API can accept under 2 MB
SUPPORTED_TEXT_SIZE = 2 * math.pow(1024, 2)

# API can accept under 120 MB
SUPPORTED_AUDIO_SIZE = 120 * math.pow(1024, 2)

# API can accept under 2GB
SUPPORTED_DICOM_SIZE = 2000 * math.pow(1024, 2)

# API can accept under 100 MB
SUPPORTED_PCD_SIZE = 100 * math.pow(1024, 2)

# API can accept under 250 MB
SUPPORTED_OBJECT_SIZE = 250 * math.pow(1024, 2)

# Only 'avc1' and 'H264' are supported for video task creation.
SUPPORTED_FOURCC = ["avc1", "h264"]


SUPPORTED_INFERENCE_IMAGE_SIZE = 6 * math.pow(1024, 2)

AttributeValue = Union[str, List[str], float, List[float]]


class AnnotationType(Enum):
    bbox = "bbox"
    polygon = "polygon"
    keypoint = "keypoint"
    line = "line"
    circle = "circle"
    segmentation = "segmentation"
    classification = "classification"
    pose_estimation = "pose_estimation"


class Priority(Enum):
    none = 0
    low = 10
    medium = 20
    high = 30


class DatasetObjectType(Enum):
    none = "none"
    train = "train"
    valid = "valid"
    test = "test"

    @classmethod
    def create(cls, value: "str | DatasetObjectType") -> "DatasetObjectType":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Invalid DatasetObjectType: {value}. "
                f"types must be {[k for k in DatasetObjectType.__members__.keys()]}"
            )
