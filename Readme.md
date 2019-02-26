# Predicting Polynuclear or Mononuclear

The diagnosis of blood-based diseases often involves identifying and characterizing patient blood samples. Automated methods to detect and classify blood cell subtypes have important medical applications. Using AI, classifying a stained image of a white blood cell as either polynuclear or mononuclear. 

```
Note that Eosinophils and Neutrophils are polynuclear while Lymphocytes and Monocytes are mononuclear. 
```

- Model used - CNN
- Framework Used - pytorch,Flask.
- Dataset - Used Augmented Dataset.
- Model Trained on - aws with p3l large instance.

### CNN:

- 3 Conv Layers.
- 3 Pooling layers.
- 1 Dropout.
- 2 hidden layers.
- 4 output nodes.

### Dataset:

The dataSample.zip in this folder consists of normal image. The image used as training in the model are augmented once ( Horizontal rotation,vertical rotation,Centered Crop, resize ....)

### To Run:

#### Pre-equisites:

- Flask
- numpy
- torch
- python(Latest)

1. Open terminal and type the below command. The Flask app will be served on http://127.0.0.1:5000/

```
python run.py
```
