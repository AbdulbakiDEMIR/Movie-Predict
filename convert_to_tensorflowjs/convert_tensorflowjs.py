import subprocess

input_path = './saved_model_dir'
output_path = './final_model'

command = [
    'tensorflowjs_converter',
    '--input_format=tf_saved_model',
    '--output_format=tfjs_graph_model',
    input_path,
    output_path
]

# Komutu çalıştır
subprocess.run(command, check=True)