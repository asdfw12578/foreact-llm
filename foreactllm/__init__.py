# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from .tokenizer import Tokenizer

# Keep the package initializer lightweight. Older ActionLLM utilities imported
# generator/model symbols here, but the SDR training path imports its concrete
# model module directly (e.g. model). Importing the legacy generator
# here would require foreactllm/model.py, which is not part of this code branch.
LaVIN_Generator = None


