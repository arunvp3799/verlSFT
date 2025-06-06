# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union, Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging
from ..template_manager import TemplateManager

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(
        self,
        parquet_files: List[str],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
    ):
        """Initialize SFT dataset.
        
        Args:
            parquet_files: List of parquet files containing training data
            tokenizer: Tokenizer to use
            config: Configuration dictionary
        """
        self.tokenizer = tokenizer
        self.config = config
        
        # Load data
        if hasattr(parquet_files, '__iter__') and not isinstance(parquet_files, str):
            # Handle OmegaConf ListConfig or other iterable types
            if hasattr(parquet_files, 'startswith'):
                # This is likely a string-like object
                parquet_files = [parquet_files]
            elif len(parquet_files) > 0:
                # Check if it's a nested list structure
                if hasattr(parquet_files[0], '__iter__') and not isinstance(parquet_files[0], str):
                    parquet_files = list(parquet_files[0])
                else:
                    parquet_files = list(parquet_files)
        else:
            # Convert single string to list
            parquet_files = [parquet_files]
        
        parquet_files_local = self._download(parquet_files)
        dataframes = []
        for f in parquet_files_local:
            df = pd.read_parquet(f)
            dataframes.append(df)
        
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
        else:
            self.data = pd.DataFrame()


        # Initialize template manager if using templates
        self.template_manager = None
        if config.get('use_template', False):
            template_path = config.get('template_path', 'verl/trainer/config/prompt_template.yaml')
            self.template_manager = TemplateManager(template_path)
            self.template_name = config.get('template_name', 'default')
            
        # Validate configuration
        self._validate_config()
    
    def _download(self, parquet_files: List[str]):
        parquet_files_local = []
        for i, parquet_file in enumerate(parquet_files):
            parquet_files_local.append(copy_to_local(parquet_file, verbose=True, use_shm=self.config.get('use_shm', False)))
        return parquet_files_local
        
    def _validate_config(self):
        """Validate dataset configuration."""
        if self.config.get('use_template', False):
            if self.config.get('use_model_chat_template', False):
                raise ValueError("Cannot use both custom template and model chat template")
                
            if not self.template_manager:
                raise ValueError("Template manager not initialized")
                
            # Validate template exists
            self.template_manager.get_template(self.template_name)
                
        elif self.config.get('use_model_chat_template', False):
            if not hasattr(self.tokenizer, 'apply_chat_template'):
                raise ValueError("Tokenizer does not support chat templates")
                
        # Validate required columns exist
        required_cols = []
        if self.config.get('use_template', False):
            # For templates, we need to validate the data matches template requirements
            template = self.template_manager.get_template(self.template_name)
            format_str = template['format']
            if '{%' in format_str or '{{' in format_str:
                # For Jinja2 templates, we can't easily extract required columns
                # Just try to format first row and catch errors
                try:
                    self.template_manager.format_prompt(self.template_name, self.data.iloc[0].to_dict())
                except Exception as e:
                    raise ValueError(f"Template validation failed: {e}")
            else:
                # For Python string formatting, extract required columns
                import string
                formatter = string.Formatter()
                required_cols = [arg[1] for arg in formatter.parse(format_str) if arg[1] is not None]
        else:
            # For standard format, check prompt and response keys
            required_cols = [
                self.config['prompt_key'],
                self.config['response_key']
            ]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")
            
    def _format_prompt(self, row: pd.Series) -> Dict[str, Any]:
        """Format prompt from row data.
        
        Returns:
            Dictionary containing:
            - text: The formatted text
            - prompt_length: Length of the prompt part
            - response_length: Length of the response part
        """
        if self.config.get('use_template', False):
            text = self.template_manager.format_prompt(self.template_name, row.to_dict())
            # For templates, we need to find where the response starts
            # This is more complex and depends on the template format
            # For now, we'll use a simple heuristic
            response_start = len(self.tokenizer.encode(text.split('\n')[-1]))
            prompt_length = len(self.tokenizer.encode(text)) - response_start
            return {
                'text': text,
                'prompt_length': prompt_length,
                'response_length': response_start
            }
        elif self.config.get('use_model_chat_template', False):
            # Convert row to chat format
            messages = []
            if self.config.get('multiturn', {}).get('enable', False):
                messages = row[self.config['multiturn']['messages_key']]
            else:
                messages = [
                    {'role': 'user', 'content': row[self.config['prompt_key']]},
                    {'role': 'assistant', 'content': row[self.config['response_key']]}
                ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            # For chat template, we need to find where the response starts
            response_start = len(self.tokenizer.encode(text.split('\n')[-1]))
            prompt_length = len(self.tokenizer.encode(text)) - response_start
            return {
                'text': text,
                'prompt_length': prompt_length,
                'response_length': response_start
            }
        else:
            # Handle prompt/response with dict access
            prompt_key = self.config['prompt_key']
            response_key = self.config['response_key']
            
            # Extract prompt and response, handling nested dictionaries
            if 'prompt_dict_keys' in self.config:
                prompt_data = row[prompt_key]
                if isinstance(prompt_data, dict):
                    prompt_values = []
                    for key in self.config['prompt_dict_keys']:
                        if key in prompt_data:
                            prompt_values.append(str(prompt_data[key]))
                    prompt = ' '.join(prompt_values)
                else:
                    prompt = str(prompt_data)
            else:
                prompt = str(row[prompt_key])
                
            if 'response_dict_keys' in self.config:
                response_data = row[response_key]
                if isinstance(response_data, dict):
                    response_values = []
                    for key in self.config['response_dict_keys']:
                        if key in response_data:
                            response_values.append(str(response_data[key]))
                    response = ' '.join(response_values)
                else:
                    response = str(response_data)
            else:
                response = str(row[response_key])
            
            # Apply chat template to prompt
            prompt_chat = [{"role": "user", "content": prompt}]
            prompt_chat_str = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            response_chat_str = response + self.tokenizer.eos_token
            
            # Get lengths for loss mask
            prompt_ids = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_ids = self.tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            
            return {
                'text': prompt_chat_str + response_chat_str,
                'prompt_length': len(prompt_ids),
                'response_length': len(response_ids)
            }
            
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item from dataset."""
        row = self.data.iloc[idx]
        formatted = self._format_prompt(row)
        # Tokenize
        encodings = self.tokenizer(
            formatted['text'],
            return_tensors='pt',
            add_special_tokens=False
        )
        
        input_ids = encodings['input_ids'][0]
        attention_mask = encodings['attention_mask'][0]
        
        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.config['max_length']:
            padded_input_ids = torch.ones(size=(self.config['max_length'] - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.config['max_length'] - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.config['max_length']:
            truncation = self.config.get('truncation', 'right')
            if truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.config['max_length'] :]
                attention_mask = attention_mask[-self.config['max_length'] :]
            elif truncation == "right":
                input_ids = input_ids[: self.config['max_length']]
                attention_mask = attention_mask[: self.config['max_length']]
            elif truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.config['max_length']=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {truncation}")
        
        # Create loss mask
        loss_mask = torch.zeros_like(input_ids)
        prompt_length = formatted['prompt_length']
        response_length = formatted['response_length']
        
        # Mask out prompt for SFT
        if prompt_length > 1:
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # Mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        
        position_ids = compute_position_id_with_mask(attention_mask)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }
        
    def __len__(self) -> int:
        return len(self.data)
