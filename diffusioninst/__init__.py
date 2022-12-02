"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

Modified by Zhangxuan Gu, Haoxing Chen
Date: Nov 30, 2022
Contact: {guzhangxuan.gzx, chenhaoxing.chx}@antgroup.com
"""

from .config import add_diffusioninst_config
from .detector import DiffusionInst
from .dataset_mapper import DiffusionInstDatasetMapper
from .test_time_augmentation import DiffusionInstWithTTA
from .swintransformer import build_swintransformer_fpn_backbone
