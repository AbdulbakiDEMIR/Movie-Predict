# TensorFlow.js Model Conversion Notes

## Installation

1. To install the required packages, run the following command:

```bash
pip install -r requirements.txt --no-deps
```

or in your python environment on windows:

````bash
python -m venv tfjs_env
tfjs_env\Scripts\activate
````

or in your python environment on macOS/Linux:

```bash
python -m venv tfjs_env
source tfjs_env/bin/activate
```

or in your conda environment:

```bash
conda activate tfjs_env
pip install -r requirements.txt
```

## Error and Solution During Conversion

On some systems, you may encounter the following error when running the `tensorflowjs_converter` command:

```
ModuleNotFoundError: No module named 'tensorflow_decision_forests'
```

The reason for this error is an unnecessary `import tensorflow_decision_forests` line in one of the files inside the `tensorflowjs` package.

### Solution

You need to locate the following file:

```
<your_python_env_path>/lib/site-packages/tensorflowjs/converters/tf_saved_model_conversion_v2.py
```

> **Note:** This file path may vary depending on your operating system, Python or conda environment name, and its location. For example, on Windows with a conda environment, the path might look like:
>
> ```
> C:\Users\your_username\.conda\envs\your_env\lib\site-packages\tensorflowjs\converters\tf_saved_model_conversion_v2.py
> ```
>
> Or, for a pip-installed environment:
>
> ```
> C:\Users\your_username\AppData\Local\Programs\Python\Python3x\Lib\site-packages\tensorflowjs\converters\tf_saved_model_conversion_v2.py
> ```
>
> **To find the exact path in your environment:**
>
> 1. Activate your Python environment.
> 2. Run the following command in a Python terminal:
>    ```python
>    import tensorflowjs
>    print(tensorflowjs.__file__)
>    ```
> 3. In the resulting path, locate the `converters/tf_saved_model_conversion_v2.py` file.

Then, find the following line:

```python
import tensorflow_decision_forests
```

**Either delete this line or comment it out by adding a `#` at the beginning:**

```python
# import tensorflow_decision_forests
```

Save the file and restart the conversion process.

---

This will resolve the error encountered during conversion. If you face any other issues, please let us know.
