# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING
from transformers.utils.file_utils import _LazyModule


_import_structure = {
    "configuration_roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig", "RobertaOnnxConfig"],
    "tokenization_roberta": ["RobertaTokenizer"],
}

_import_structure["tokenization_roberta_fast"] = ["RobertaTokenizerFast"]
_import_structure["modeling_roberta"] = [
    "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
    "RobertaForCausalLM",
    "RobertaForMultipleChoice",
    "RobertaForTokenClassification",
    "RobertaForQuestionAnswering",
    "RobertaForSequenceClassification",
    "RobertaModel",
    "RobertaForMaskedLM",
    "RobertaPreTrainedModel",
]


if TYPE_CHECKING:
    from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig, RobertaOnnxConfig
    from .tokenization_roberta import RobertaTokenizer

    from .tokenization_roberta_fast import RobertaTokenizerFast

    from .modeling_roberta import (
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        RobertaForCausalLM,
        RobertaForMultipleChoice,
        RobertaForTokenClassification,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaModel,
        RobertaForMaskedLM,
        RobertaPreTrainedModel,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
