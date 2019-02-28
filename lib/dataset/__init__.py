# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .coco_sam import COCOSAMDataset as coco_sam
from .coco_fields import COCOFieldsDataset as coco_fields
from .coco_bc import COCOBCDataset as coco_bc