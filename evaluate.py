import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

model = load_model('deepfake_detector.keras')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    'dataset/test/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

y_pred_prob = model.predict(test_generator, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:\n")
print(report)
