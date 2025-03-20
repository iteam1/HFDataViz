#!/usr/bin/env python
"""
HuggingFace Dataset Visualizer - A simple standalone tool to visualize HuggingFace datasets
without requiring a web server.

This script:
1. Loads a HuggingFace dataset specified by the user
2. Displays dataset information
3. Shows examples in a formatted way
4. Allows navigation through examples

Usage:
    python dataset_visualizer.py [dataset_name] [config_name]
    
Example:
    python dataset_visualizer.py HuggingFaceTB/smoltalk all
"""

import sys
import json
from datasets import load_dataset, get_dataset_config_names
import os
import textwrap
from typing import Dict, Any, List, Optional, Union

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_value(value: Any, indent: int = 0, max_width: int = 80) -> str:
    """Format a value for display, handling different types appropriately."""
    indent_str = ' ' * indent
    
    if isinstance(value, (dict, list)):
        try:
            # For dictionaries and lists, use pretty JSON formatting
            formatted = json.dumps(value, indent=2, ensure_ascii=False)
            # Add indentation to each line
            return '\n'.join(indent_str + line for line in formatted.splitlines())
        except (TypeError, ValueError):
            # Fall back to str if JSON serialization fails
            return str(value)
    elif isinstance(value, str):
        if len(value) > max_width:
            # For long strings, wrap them
            wrapped = textwrap.fill(value, width=max_width - indent)
            # Add indentation to each line
            return '\n'.join(indent_str + line for line in wrapped.splitlines())
        return value
    else:
        return str(value)

def display_example(example: Dict[str, Any], index: int, total: int):
    """Display a single example in a formatted way."""
    clear_screen()
    
    # Header
    print(f"{Colors.HEADER}{Colors.BOLD}🤗 Example {index + 1} of {total} {Colors.ENDC}\n")
    
    term_width = os.get_terminal_size().columns
    
    # Display each field in the example
    for key, value in example.items():
        print(f"{Colors.BOLD}{Colors.CYAN}{key}:{Colors.ENDC}")
        
        # Special handling for message content
        if key in ['content', 'message', 'text', 'prompt', 'completion'] and isinstance(value, str):
            # For message content fields, use special formatting
            print(f"{Colors.BLUE}┌{'─' * (term_width - 2)}┐{Colors.ENDC}")
            
            # Format and wrap the text
            wrapped_text = format_value(value, indent=4, max_width=term_width-8)
            lines = wrapped_text.splitlines()
            
            for line in lines:
                print(f"{Colors.BLUE}│{Colors.ENDC} {Colors.GREEN}{line}{' ' * (term_width - len(line) - 4)}{Colors.BLUE}│{Colors.ENDC}")
            
            print(f"{Colors.BLUE}└{'─' * (term_width - 2)}┘{Colors.ENDC}")
        elif isinstance(value, (dict, list)):
            # For complex objects, pretty print with json
            print(f"{Colors.YELLOW}{format_value(value, indent=2, max_width=term_width-4)}{Colors.ENDC}")
        elif isinstance(value, str) and len(value) > 100:
            # For long strings, format with wrapping
            print(f"{Colors.GREEN}{format_value(value, indent=2, max_width=term_width-4)}{Colors.ENDC}")
        else:
            # For simple values
            print(f"{Colors.GREEN}{value}{Colors.ENDC}")
        
        print()  # Add line between fields

def display_help():
    """Display navigation help."""
    print(f"\n{Colors.BOLD}Navigation:{Colors.ENDC}")
    print(f"  {Colors.BLUE}n{Colors.ENDC} - Next example")
    print(f"  {Colors.BLUE}p{Colors.ENDC} - Previous example")
    print(f"  {Colors.BLUE}j NUMBER{Colors.ENDC} - Jump to example number")
    print(f"  {Colors.BLUE}i{Colors.ENDC} - Dataset info")
    print(f"  {Colors.BLUE}q{Colors.ENDC} - Quit")
    print(f"  {Colors.BLUE}h{Colors.ENDC} - Show this help")

def display_dataset_info(dataset, dataset_name: str, config_name: str):
    """Display information about the dataset."""
    clear_screen()
    
    # Header with emoji
    print(f"{Colors.HEADER}{Colors.BOLD}🤗 HuggingFace Dataset Visualizer {Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Dataset:{Colors.ENDC} {Colors.GREEN}{dataset_name}{Colors.ENDC}")
    print(f"{Colors.BOLD}Config:{Colors.ENDC} {Colors.GREEN}{config_name}{Colors.ENDC}")
    print(f"{Colors.BOLD}Number of examples:{Colors.ENDC} {Colors.GREEN}{len(dataset)}{Colors.ENDC}")
    
    # Display features
    print(f"\n{Colors.BOLD}Features:{Colors.ENDC}")
    for feature_name, feature_type in dataset.features.items():
        print(f"  {Colors.CYAN}{feature_name}{Colors.ENDC}: {Colors.YELLOW}{feature_type}{Colors.ENDC}")
    
    print("\nPress any key to view examples...")

def prompt_for_config(dataset_name: str) -> str:
    """Prompt the user to select a config for the dataset."""
    try:
        configs = get_dataset_config_names(dataset_name)
        
        if not configs:
            return None  # No configs available
            
        print(f"\n{Colors.BOLD}Available configs for {dataset_name}:{Colors.ENDC}")
        for i, config in enumerate(configs):
            print(f"  {i+1}. {Colors.CYAN}{config}{Colors.ENDC}")
            
        while True:
            try:
                choice = input(f"\n{Colors.BOLD}Select a config (1-{len(configs)}) or enter name directly:{Colors.ENDC} ")
                
                # Check if input is a number
                if choice.isdigit() and 1 <= int(choice) <= len(configs):
                    return configs[int(choice) - 1]
                
                # Check if input is a valid config name
                if choice in configs:
                    return choice
                
                print(f"{Colors.RED}Invalid selection. Please try again.{Colors.ENDC}")
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid input. Please try again.{Colors.ENDC}")
                
    except Exception as e:
        print(f"{Colors.RED}Error fetching config names: {str(e)}{Colors.ENDC}")
        return None

def main():
    """Main function to run the visualizer."""
    # Check for arguments
    if len(sys.argv) < 2:
        print(f"{Colors.YELLOW}🤗 HuggingFace Dataset Visualizer {Colors.ENDC}")
        print(f"\n{Colors.BOLD}Usage:{Colors.ENDC}")
        print(f"  python dsviz.py [dataset_name] [config_name]")
        print(f"\n{Colors.BOLD}Example:{Colors.ENDC}")
        print(f"  python dsviz.py HuggingFaceTB/smoltalk all")
        return
    
    # Get dataset name from command line or prompt
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        print(f"{Colors.BOLD}Enter dataset name (e.g., HuggingFaceTB/smoltalk):{Colors.ENDC} ", end="")
        dataset_name = input().strip()
    
    if not dataset_name:
        print(f"{Colors.RED}Error: No dataset name provided.{Colors.ENDC}")
        return

    # Get config name from command line or prompt
    config_name = None
    if len(sys.argv) > 2:
        config_name = sys.argv[2]
    
    # Load dataset
    print(f"\n{Colors.YELLOW}Loading dataset {dataset_name}...{Colors.ENDC}")
    try:
        # First try to load without config
        try:
            dataset = load_dataset(dataset_name, split="train")
            print(f"{Colors.GREEN}Dataset loaded successfully!{Colors.ENDC}")
        except ValueError as e:
            # If config is needed but not provided, prompt user
            if "Config name is missing" in str(e) and not config_name:
                print(f"{Colors.YELLOW}This dataset requires a config name.{Colors.ENDC}")
                config_name = prompt_for_config(dataset_name)
                
                if not config_name:
                    print(f"{Colors.RED}No config selected. Exiting.{Colors.ENDC}")
                    return
                
                dataset = load_dataset(dataset_name, config_name, split="train")
                print(f"{Colors.GREEN}Dataset loaded successfully with config '{config_name}'!{Colors.ENDC}")
            else:
                raise e
    except Exception as e:
        print(f"{Colors.RED}Error loading dataset: {str(e)}{Colors.ENDC}")
        return
    
    # Start visualization
    current_index = 0
    total_examples = len(dataset)
    
    while True:
        # Display current example
        display_example(dataset[current_index], current_index, total_examples)
        display_help()
        
        # Get user input
        cmd = input(f"\n{Colors.BOLD}Enter command:{Colors.ENDC} ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'n':
            if current_index < total_examples - 1:
                current_index += 1
        elif cmd == 'p':
            if current_index > 0:
                current_index -= 1
        elif cmd.startswith('j '):
            try:
                jump_to = int(cmd.split(' ')[1]) - 1  # Convert to 0-indexed
                if 0 <= jump_to < total_examples:
                    current_index = jump_to
                else:
                    print(f"{Colors.RED}Invalid index. Must be between 1 and {total_examples}.{Colors.ENDC}")
                    input("Press Enter to continue...")
            except ValueError:
                print(f"{Colors.RED}Invalid number.{Colors.ENDC}")
                input("Press Enter to continue...")
        elif cmd == 'i':
            display_dataset_info(dataset, dataset_name, config_name or "default")
        elif cmd == 'h':
            # Help is already displayed after each example
            pass
        else:
            print(f"{Colors.RED}Unknown command.{Colors.ENDC}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
