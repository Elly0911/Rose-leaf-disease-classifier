"""
Educational descriptions for each rose leaf class.

The five class names must match the order used when training the
model. Update CLASS_NAMES if your model was trained on a different
label ordering.
"""

DISEASE_INFO = {
    "Black Spot": {
        "title": "Black Spot",
        "description": (
            "Black spot is a fungal disease of roses characterised by "
            "dark, circular spots on the leaves. Infected leaves may turn "
            "yellow and fall prematurely, weakening the plant."
        ),
        "cause": "Fungal infection, commonly caused by Diplocarpon rosae.",
        "symptoms": [
            "Dark, circular spots on leaves",
            "Yellowing around infected areas",
            "Premature leaf drop",
        ],
        "recommendation": (
            "Remove infected leaves, avoid wetting the foliage when watering, "
            "improve air circulation, and consider an appropriate fungicide "
            "when necessary."
        ),
    },

    "Downy Mildew": {
        "title": "Downy Mildew",
        "description": (
            "Downy mildew is a disease that can affect rose foliage, "
            "causing discoloured lesions and, under favourable conditions, "
            "a downy growth on the undersides of leaves."
        ),
        "cause": (
            "An oomycete pathogen, commonly associated with "
            "Peronospora sparsa in roses."
        ),
        "symptoms": [
            "Purple, reddish, or brown leaf lesions",
            "Yellowing or discolouration",
            "Downy growth on leaf undersides in suitable conditions",
        ],
        "recommendation": (
            "Remove infected plant material, reduce prolonged leaf wetness, "
            "improve air circulation, and use an appropriate disease-control "
            "product if required."
        ),
    },

    "Fresh Leaf": {
        "title": "Fresh Leaf",
        "description": (
            "This category represents rose leaves classified by the model "
            "as fresh or apparently healthy, without the characteristic "
            "visual symptoms represented by the disease classes."
        ),
        "cause": "No visible disease is indicated by this classification.",
        "symptoms": [
            "Generally uniform leaf colour",
            "No obvious characteristic disease lesions",
            "No clear signs of substantial leaf damage",
        ],
        "recommendation": (
            "Continue routine plant care and monitor the leaves regularly "
            "for any changes or emerging symptoms."
        ),
    },

    "Insect Hole": {
        "title": "Insect Hole",
        "description": (
            "This category represents rose leaves with visible holes or "
            "feeding damage that may be associated with insects or other "
            "leaf-feeding pests."
        ),
        "cause": (
            "Possible feeding activity by leaf-chewing insects or other "
            "pests; the image alone cannot identify the specific pest."
        ),
        "symptoms": [
            "Holes in the leaf blade",
            "Irregular or chewed leaf margins",
            "Visible tissue loss",
        ],
        "recommendation": (
            "Inspect the plant and the undersides of leaves for pests, "
            "remove severely damaged foliage when appropriate, and select "
            "a suitable pest-management method if an infestation is confirmed."
        ),
    },

    "ROSE_MOSAIC": {
        "title": "Rose Mosaic",
        "description": (
            "Rose mosaic refers to a group of viral diseases that can cause "
            "mosaic-like patterns, mottling, yellow lines, or irregular "
            "discolouration on rose leaves."
        ),
        "cause": (
            "Viral infection; several viruses may be associated with "
            "rose mosaic symptoms."
        ),
        "symptoms": [
            "Mottled or mosaic-like leaf patterns",
            "Yellow lines or irregular discolouration",
            "Variable leaf symptoms depending on the infection",
        ],
        "recommendation": (
            "Seek confirmation if viral disease is suspected. Use healthy "
            "planting material, disinfect pruning tools, and manage affected "
            "plants according to local horticultural guidance."
        ),
    },
}

# Ordered list matching the model's output index.
CLASS_NAMES = [
    "Black Spot",
    "Downy Mildew",
    "Fresh Leaf",
    "Insect Hole",
    "ROSE_MOSAIC",
]

# Display names for UI — keeps internal class keys stable while
# presenting human-friendly labels to the user.
DISPLAY_NAMES = {
    "Black Spot": "Black Spot",
    "Downy Mildew": "Downy Mildew",
    "Fresh Leaf": "Fresh Leaf",
    "Insect Hole": "Insect Hole",
    "ROSE_MOSAIC": "Rose Mosaic",
}


def display_name(class_name):
    """Return the human-friendly display name for a class key."""
    return DISPLAY_NAMES.get(class_name, class_name)


def get_disease_info(class_name):
    """Return info dict for a class name, with a safe fallback."""
    info = DISEASE_INFO.get(class_name)
    if info is None:
        return {
            "title": display_name(class_name),
            "description": (
                "No additional information is available for this class."
            ),
            "cause": "Unknown.",
            "symptoms": [],
            "recommendation": (
                "Consult a plant-disease expert for guidance."
            ),
        }
    return info