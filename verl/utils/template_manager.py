import os
import yaml
from typing import Dict, Any, Optional
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

class TemplateManager:
    def __init__(self, template_path: str):
        """Initialize template manager with path to template config file.
        
        Args:
            template_path: Path to the prompt_template.yaml file
        """
        self.template_path = template_path
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load templates from YAML file."""
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Template file not found at {self.template_path}")
            
        with open(self.template_path, 'r') as f:
            templates = yaml.safe_load(f)
            
        if not isinstance(templates, dict):
            raise ValueError("Template file must contain a dictionary of templates")
            
        # Validate template format
        for name, template in templates.items():
            if not isinstance(template, dict) or 'format' not in template:
                raise ValueError(f"Template '{name}' must have a 'format' field")
                
        return templates
        
    def get_template(self, template_name: str) -> Optional[Dict[str, str]]:
        """Get template by name."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found in {self.template_path}")
        return self.templates[template_name]
        
    def format_prompt(self, template_name: str, data: Dict[str, Any]) -> str:
        """Format prompt using specified template and data.
        
        Args:
            template_name: Name of template to use
            data: Dictionary of data to format template with
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(template_name)
        format_str = template['format']
        
        try:
            # Try Jinja2 templating first (for multi-turn chat)
            if '{%' in format_str or '{{' in format_str:
                return Template(format_str).render(**data)
            # Fall back to Python string formatting
            return format_str.format(**data)
        except KeyError as e:
            raise ValueError(f"Missing required template argument: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template: {e}")
            
    def validate_template_data(self, template_name: str, data: Dict[str, Any]) -> bool:
        """Validate that data contains all required template arguments.
        
        Args:
            template_name: Name of template to validate against
            data: Dictionary of data to validate
            
        Returns:
            True if data is valid, raises ValueError if not
        """
        template = self.get_template(template_name)
        format_str = template['format']
        
        # Extract required arguments from format string
        if '{%' in format_str or '{{' in format_str:
            # For Jinja2 templates, we can't easily extract required args
            # Just try to render and catch errors
            try:
                Template(format_str).render(**data)
                return True
            except Exception as e:
                raise ValueError(f"Template validation failed: {e}")
        else:
            # For Python string formatting, extract required args
            import string
            formatter = string.Formatter()
            required_args = [arg[1] for arg in formatter.parse(format_str) if arg[1] is not None]
            
            missing_args = [arg for arg in required_args if arg not in data]
            if missing_args:
                raise ValueError(f"Missing required template arguments: {missing_args}")
                
        return True 