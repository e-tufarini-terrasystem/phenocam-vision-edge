"""
Define the fixed, version-controlled COCO class configuration.

Names, grouping, order, and tuple structure are invariants. Operators may
modify only the boolean value attached to each class.

The committed defaults enable people and selected road vehicles for privacy analysis.
"""

COCO_CLASSES = (
    ("person", (
        ("person", True),
    )),
    ("vehicle", (
        ("bicycle", False),
        ("car", True),
        ("motorcycle", True),
        ("airplane", False),
        ("bus", True),
        ("train", False),
        ("truck", True),
        ("boat", False),
    )),
    ("outdoor", (
        ("traffic light", False),
        ("fire hydrant", False),
        ("stop sign", False),
        ("parking meter", False),
        ("bench", False),
    )),
    ("animal", (
        ("bird", False),
        ("cat", False),
        ("dog", False),
        ("horse", False),
        ("sheep", False),
        ("cow", False),
        ("elephant", False),
        ("bear", False),
        ("zebra", False),
        ("giraffe", False),
    )),
    ("accessory", (
        ("backpack", False),
        ("umbrella", False),
        ("handbag", False),
        ("tie", False),
        ("suitcase", False),
    )),
    ("sports", (
        ("frisbee", False),
        ("skis", False),
        ("snowboard", False),
        ("sports ball", False),
        ("kite", False),
        ("baseball bat", False),
        ("baseball glove", False),
        ("skateboard", False),
        ("surfboard", False),
        ("tennis racket", False),
    )),
    ("kitchen", (
        ("bottle", False),
        ("wine glass", False),
        ("cup", False),
        ("fork", False),
        ("knife", False),
        ("spoon", False),
        ("bowl", False),
    )),
    ("food", (
        ("banana", False),
        ("apple", False),
        ("sandwich", False),
        ("orange", False),
        ("broccoli", False),
        ("carrot", False),
        ("hot dog", False),
        ("pizza", False),
        ("donut", False),
        ("cake", False),
    )),
    ("furniture", (
        ("chair", False),
        ("couch", False),
        ("potted plant", False),
        ("bed", False),
        ("dining table", False),
        ("toilet", False),
    )),
    ("electronic", (
        ("tv", False),
        ("laptop", False),
        ("mouse", False),
        ("remote", False),
        ("keyboard", False),
        ("cell phone", False),
    )),
    ("appliance", (
        ("microwave", False),
        ("oven", False),
        ("toaster", False),
        ("sink", False),
        ("refrigerator", False),
    )),
    ("indoor", (
        ("book", False),
        ("clock", False),
        ("vase", False),
        ("scissors", False),
        ("teddy bear", False),
        ("hair drier", False),
        ("toothbrush", False),
    )),
)
