from flask import Flask, render_template, request, jsonify
from datasets import load_dataset, get_dataset_config_names
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_configs', methods=['POST'])
def api_get_configs():
    data = request.json
    dataset_name = data.get('dataset_name', '')
    
    try:
        configs = get_dataset_config_names(dataset_name)
        return jsonify({
            'success': True,
            'configs': configs
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/load_dataset', methods=['POST'])
def api_load_dataset():
    data = request.json
    dataset_name = data.get('dataset_name', '')
    config_name = data.get('config_name', None)
    
    try:
        # Load the dataset
        try:
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split='train')
            else:
                dataset = load_dataset(dataset_name, split='train')
        except ValueError as e:
            # If config is missing, return available configs
            if "Config name is missing" in str(e):
                configs = get_dataset_config_names(dataset_name)
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'need_config': True,
                    'configs': configs,
                    'message': '🔍 This dataset requires a configuration!'
                })
            else:
                raise e
        
        # Get dataset information
        info = {
            'num_examples': len(dataset),
            'features': list(dataset.features.keys()),
            'dataset_name': dataset_name,
            'config_name': config_name
        }
        
        # Cache the first few examples
        examples = []
        for i in range(min(10, len(dataset))):
            example = dataset[i]
            # Convert any special types to strings for JSON serialization
            serializable_example = {}
            for key, value in example.items():
                # Special handling for message-type fields
                if key in ['content', 'message', 'text', 'prompt', 'completion'] and isinstance(value, str):
                    # Preserve messages exactly as they are
                    serializable_example[key] = value
                elif hasattr(value, 'tolist'):  # For numpy arrays
                    serializable_example[key] = value.tolist()
                else:
                    serializable_example[key] = str(value) if not isinstance(value, (int, float, bool, str, list, dict)) else value
            examples.append(serializable_example)
        
        return jsonify({
            'success': True,
            'info': info,
            'examples': examples
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/get_example', methods=['POST'])
def api_get_example():
    data = request.json
    dataset_name = data.get('dataset_name', '')
    config_name = data.get('config_name', None)
    index = data.get('index', 0)
    
    try:
        # Load the dataset
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split='train')
        else:
            dataset = load_dataset(dataset_name, split='train')
        
        if index < 0 or index >= len(dataset):
            return jsonify({
                'success': False,
                'error': f"Index {index} out of range (0-{len(dataset)-1})"
            })
        
        # Get the example
        example = dataset[index]
        
        # Convert any special types to strings for JSON serialization
        serializable_example = {}
        for key, value in example.items():
            # Special handling for message-type fields
            if key in ['content', 'message', 'text', 'prompt', 'completion'] and isinstance(value, str):
                # Preserve messages exactly as they are
                serializable_example[key] = value
            elif hasattr(value, 'tolist'):  # For numpy arrays
                serializable_example[key] = value.tolist()
            else:
                serializable_example[key] = str(value) if not isinstance(value, (int, float, bool, str, list, dict)) else value
        
        return jsonify({
            'success': True,
            'example': serializable_example,
            'index': index,
            'total': len(dataset)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
