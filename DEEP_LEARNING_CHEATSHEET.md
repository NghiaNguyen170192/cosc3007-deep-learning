# Deep Learning Cheatsheet
**COSC3007 - Deep Learning Course Summary**  
**Based on:** _Deep Learning with Python_ by François Chollet

---

## Week 1-2: Mathematical Building Blocks & Frameworks

### Tensors
**Tensor**: Multi-dimensional array (container for numerical data)
- **Scalar** (0D tensor): Single number
- **Vector** (1D tensor): Array of numbers
- **Matrix** (2D tensor): 2D array
- **3D+ tensor**: Higher-dimensional arrays

**Tensor Attributes**:
- **Rank/Ndim**: Number of axes (dimensions)
- **Shape**: Tuple describing size along each axis
- **dtype**: Data type (float32, int32, etc.)

### Tensor Operations
**Element-wise operations**: Applied independently to each entry
- Addition, multiplication, ReLU, etc.
- Broadcasting: Automatic expansion of smaller tensor

**Dot product**: `(a, b, c, d) · (d, e) → (a, b, c, e)`
- Combines entries from input tensors
- Matrix multiplication is dot product

**Tensor reshaping**:
- `reshape()`: Change shape without modifying data
- `transpose()`: Swap axes

### Gradient-Based Optimization
**Derivative**: Slope of continuous function at a point
- Describes how output changes when input changes

**Gradient**: Vector derivative for multi-input functions
- Points in direction of steepest ascent
- **Gradient descent**: Move opposite to gradient to minimize loss

**Backpropagation**: 
- Efficient algorithm to compute gradients in neural networks
- Uses chain rule to propagate derivatives backward through layers

---

## Week 3: Neural Networks Fundamentals

### Core Components

**Layer**: Data processing module that transforms input
- Extracts representations useful for solving the problem
- Most layers have **weights** (learned parameters)

**Model**: Network of layers (directed acyclic graph)
- **Sequential**: Linear stack of layers
- **Functional API**: More flexible graph structures

**Loss Function** (Objective function):
- Measures how well network performs on training data
- Quantity to minimize during training
- Examples: MSE (regression), cross-entropy (classification)

**Optimizer**: Algorithm to update weights based on gradients
- **SGD** (Stochastic Gradient Descent)
- **Adam**: Adaptive learning rate optimizer
- **RMSprop**: Good for recurrent networks

**Metrics**: Measure to monitor during training/testing
- Accuracy, precision, recall, MAE, etc.

### Training Process

**Epoch**: One pass through entire training dataset

**Batch**: Subset of training data processed together
- **Batch size**: Number of samples per batch
- **Mini-batch SGD**: Update weights after each batch

**Learning rate**: Step size for gradient descent
- Too high: Unstable training
- Too low: Slow convergence

**Validation**: Evaluating model on held-out data
- Prevents overfitting
- Guides model selection

---

## Week 3-4: Classification & Regression

### Binary Classification
**Sigmoid activation**: `σ(x) = 1/(1+e^(-x))`
- Outputs probability between 0 and 1
- Used in final layer for binary classification

**Binary cross-entropy loss**:
- Measures distance between probability distributions
- Appropriate for binary labels

### Multi-class Classification
**Softmax activation**: Normalizes outputs to probability distribution
- All outputs sum to 1
- Used in final layer for multi-class classification

**Categorical cross-entropy**:
- For one-hot encoded labels
- `categorical_crossentropy`

**Sparse categorical cross-entropy**:
- For integer labels
- `sparse_categorical_crossentropy`

### Regression
**Linear activation** (no activation):
- Final layer outputs continuous values
- Default for regression problems

**MSE (Mean Squared Error)**:
- Standard loss for regression
- Penalizes large errors more

**MAE (Mean Absolute Error)**:
- Alternative regression loss
- More robust to outliers

---

## Week 4: ML Fundamentals & Best Practices

### Data Preprocessing

**Normalization**: Scale features to similar ranges
- Subtract mean, divide by std: `(x - mean) / std`
- Or scale to [0, 1]: `(x - min) / (max - min)`

**Vectorization**: Convert data to tensors
- Images: Normalize pixel values [0, 1] or [-1, 1]
- Text: One-hot encoding, word embeddings

**Handling missing values**:
- Fill with 0, mean, or median
- Or use special category for categorical data

### Train/Validation/Test Split

**Training set**: Data used to train model (fit weights)

**Validation set**: Tune hyperparameters, monitor overfitting
- Should NOT be used for training

**Test set**: Final evaluation of model performance
- Only use once at the end

**K-fold cross-validation**: 
- Split data into K partitions
- Train K models, each using different fold for validation
- Average results for robust estimate

### Overfitting & Regularization

**Overfitting**: Model performs well on training but poorly on validation
- Memorizes training data instead of learning patterns

**Regularization techniques**:

1. **L1/L2 Regularization**: Add penalty to loss based on weight magnitude
   - L1: Encourages sparsity (many weights → 0)
   - L2: Encourages small weights

2. **Dropout**: Randomly drop units during training
   - Prevents co-adaptation
   - Rate 0.2-0.5 common

3. **Early stopping**: Stop training when validation loss stops improving

4. **Data augmentation**: Create new training samples
   - Random transformations (rotate, flip, crop images)

5. **Reduce model capacity**: Fewer layers or units

### Universal ML Workflow

1. **Define the problem**: Classification, regression, ranking, etc.
2. **Measure success**: Choose appropriate metric
3. **Prepare data**: Vectorization, normalization, train/val/test split
4. **Develop baseline**: Simple model to beat
5. **Scale up**: Increase capacity until overfitting
6. **Regularize**: Add dropout, L2, etc.
7. **Tune hyperparameters**: Learning rate, architecture, etc.

---

## Week 5+: Convolutional Neural Networks (CNNs)

### Convolution Operation

**Convolution**: Learns local patterns in input
- Operates on small windows (patches) of input
- Shares same transformation across all patches

**Kernel (Filter)**: Small matrix of learnable weights
- Slides across input
- Produces feature map

**Key properties**:
- **Translation invariance**: Pattern learned anywhere applies everywhere
- **Spatial hierarchy**: Learn local patterns, then combine them

### CNN Layers

**Conv2D**: 2D convolution layer
- Parameters: `filters`, `kernel_size`, `strides`, `padding`, `activation`
- Input shape: `(height, width, channels)`
- Output: `(new_height, new_width, filters)`

**Padding**:
- `'valid'`: No padding (output smaller than input)
- `'same'`: Zero-padding to preserve spatial dimensions

**Stride**: Step size when sliding kernel
- Stride > 1 downsamples the output

**Parameter count for Conv2D**:
- Formula: `(kernel_h × kernel_w × input_channels + 1) × num_filters`
- `+1` for bias term per filter

**Output size calculation (valid padding)**:
- `output_size = (input_size - kernel_size) / stride + 1`

**MaxPooling2D**: Downsampling operation
- Takes maximum value in each window
- Reduces spatial dimensions
- Provides translation invariance
- Typical pool size: 2×2

**Flatten**: Converts multi-dimensional tensor to 1D
- Needed before Dense layers
- Example: (7, 7, 64) → (3136,)

**GlobalAveragePooling2D**: Average entire feature map to single value
- Alternative to Flatten + Dense
- Reduces overfitting (fewer parameters)

### CNN Architecture Patterns

**Basic pattern**:
```
Input → (Conv2D → Activation → MaxPooling)×N → Flatten → Dense → Output
```

**Depth increases, spatial size decreases**:
- More filters in deeper layers (32 → 64 → 128)
- Spatial dimensions shrink via pooling/strides

**Residual connections (ResNet)**:
- Skip connections: `output = activation(Conv(x) + x)`
- Helps training very deep networks
- Prevents vanishing gradients

**Batch normalization**:
- Normalizes layer inputs during training
- Accelerates training, allows higher learning rates
- Provides regularization effect

---

## Week 6-7: Transfer Learning & Fine-tuning

### Transfer Learning

**Pretrained model**: Model trained on large dataset (e.g., ImageNet)
- Learned general visual features
- Can be reused for new tasks

**Feature extraction**:
- Use pretrained convolutional base
- Freeze weights (set `trainable=False`)
- Add new classifier on top
- Train only new layers

**When to use transfer learning**:
- Small dataset (can't train from scratch)
- Similar domain (e.g., natural images)

### Fine-tuning

**Fine-tuning**: Unfreezing top layers of frozen model
- Train them jointly with added classifier
- Use very small learning rate
- Adapts pretrained features to new task

**Fine-tuning steps**:
1. Add custom layers on top of pretrained base
2. Freeze base and train top layers
3. Unfreeze top layers of base
4. Train both unfrozen base layers and top layers with low learning rate

**Common pretrained models**:
- VGG16, VGG19
- ResNet50, ResNet101
- InceptionV3
- Xception
- EfficientNet

---

## Week 8: Advanced CNN Topics

### Data Augmentation

**Data augmentation**: Artificially expand training set
- Random transformations that preserve labels
- Helps prevent overfitting

**Image augmentation techniques**:
- Random rotation
- Random horizontal/vertical flip
- Random zoom
- Random crop
- Color jitter (brightness, contrast, saturation)

**Keras ImageDataGenerator**:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
```

### Model Interpretation

**Visualizing intermediate activations**:
- See what patterns each layer detects
- Earlier layers: Edges, textures
- Deeper layers: Complex patterns

**Grad-CAM (Gradient Class Activation Map)**:
- Heatmap showing which parts of image are important
- Highlights regions that contribute to prediction

**Filter visualization**:
- Generate input that maximally activates a filter
- Reveals what pattern the filter detects

---

## Week 9: Image Segmentation

### Segmentation

**Image segmentation**: Assign class label to every pixel
- **Semantic segmentation**: Same class = same label
- **Instance segmentation**: Different instances = different labels

**Output**: Same spatial size as input, with class predictions per pixel

### U-Net Architecture

**U-Net**: Popular segmentation architecture
- **Contracting path** (encoder): Downsampling with convolutions
- **Expanding path** (decoder): Upsampling to original size
- **Skip connections**: Concatenate encoder features with decoder
  - Preserves spatial information lost during downsampling

**UpSampling2D**: Increases spatial dimensions
- Repeats rows/columns

**Conv2DTranspose**: Learnable upsampling
- Reverse of convolution (also called deconvolution)

---

## Week 10: Recurrent Neural Networks (RNNs)

### Sequence Processing

**Sequence**: Ordered data where position matters
- Time series, text, audio, video

**Recurrent layer**: Processes sequences step-by-step
- Maintains internal state (memory)
- Same weights applied at each timestep

### RNN Variants

**SimpleRNN**: Basic recurrent layer
- `state_t = activation(dot(input_t, W) + dot(state_t-1, U) + b)`
- Problem: Vanishing/exploding gradients

**LSTM (Long Short-Term Memory)**:
- Designed to handle long-term dependencies
- Uses gates (forget, input, output) to control information flow
- Solves vanishing gradient problem

**GRU (Gated Recurrent Unit)**:
- Simplified LSTM (fewer parameters)
- Often similar performance to LSTM

### RNN Input/Output Shapes

**Input**: `(batch_size, timesteps, features)`
- Example: (32, 50, 128) = 32 samples, 50 timesteps, 128 features per step

**return_sequences=False**: Returns last output `(batch_size, units)`

**return_sequences=True**: Returns all outputs `(batch_size, timesteps, units)`

**Bidirectional RNN**: Processes sequence forward and backward
- Captures both past and future context
- Doubles number of parameters

---

## Week 11: Natural Language Processing (NLP)

### Text Preprocessing

**Tokenization**: Split text into words (tokens)
- Vocabulary: Set of unique tokens
- Index: Integer assigned to each token

**TextVectorization layer**:
- Converts strings to sequences of integers
- Handles vocabulary building and encoding

**Padding**: Make all sequences same length
- Add 0s to shorter sequences
- Usually at beginning or end

### Text Representation

**One-hot encoding**: Binary vector per token
- Size = vocabulary size
- Sparse, no semantic meaning

**Word embeddings**: Dense vector representation
- Learns semantic relationships
- Similar words have similar vectors
- Typical dimensions: 50-300

**Embedding layer**:
- `Embedding(vocab_size, embedding_dim)`
- Input: Integer sequences `(batch, sequence_length)`
- Output: `(batch, sequence_length, embedding_dim)`
- Learnable weights

**Pretrained embeddings**:
- Word2Vec, GloVe, FastText
- Trained on large text corpora

### Text Models

**Bag-of-words**: Count/average word vectors
- Ignores word order
- Simple baseline

**1D Convolution** (Conv1D):
- Learns local patterns in sequences
- Faster than RNNs
- Good for short sequences

**RNN models**:
- Process sequences in order
- Capture long-range dependencies
- Can be slow to train

### Transformer Architecture

**Self-attention**: Relates different positions of sequence
- Computes importance of each word to every other word
- Parallel processing (unlike RNNs)

**Multi-head attention**: Multiple attention mechanisms
- Learn different relationships

**Positional encoding**: Adds position information
- Since attention has no built-in order

**Transformer encoder**: Stack of attention + feed-forward layers
- Used in BERT, GPT, etc.
- State-of-the-art for many NLP tasks

**Key advantages**:
- Parallelizable (faster training than RNNs)
- Better at capturing long-range dependencies
- Scales well with data and compute

---

## Key Activation Functions

**ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`
- Most common for hidden layers
- Solves vanishing gradient
- Fast to compute

**Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- Outputs [0, 1]
- Binary classification final layer

**Tanh**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- Outputs [-1, 1]
- Sometimes used in RNNs

**Softmax**: `f(x_i) = e^(x_i) / Σ e^(x_j)`
- Outputs probability distribution
- Multi-class classification final layer

**Linear** (no activation): `f(x) = x`
- Regression final layer

---

## Common Loss Functions

### Classification
- **Binary cross-entropy**: `binary_crossentropy`
- **Categorical cross-entropy**: `categorical_crossentropy`
- **Sparse categorical cross-entropy**: `sparse_categorical_crossentropy`

### Regression
- **Mean Squared Error (MSE)**: `mse` or `mean_squared_error`
- **Mean Absolute Error (MAE)**: `mae` or `mean_absolute_error`
- **Huber loss**: Robust to outliers

---

## Common Optimizers

**SGD (Stochastic Gradient Descent)**:
- Simple, requires careful learning rate tuning
- Can add momentum for faster convergence

**RMSprop**:
- Adaptive learning rate per parameter
- Good for RNNs

**Adam**:
- Adaptive learning rates + momentum
- Generally good default choice
- Typical learning rate: 1e-3 to 1e-4

**Learning rate scheduling**:
- Reduce learning rate when progress stalls
- `ReduceLROnPlateau` callback
- Cosine annealing

---

## Keras API Patterns

### Sequential Model
```python
from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
```

### Functional API
```python
from tensorflow.keras import Input, Model, layers

inputs = Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

### Multi-output Model
```python
model = Model(
    inputs=inputs,
    outputs={'output_A': out_A, 'output_B': out_B}
)

model.compile(
    loss={'output_A': 'mse', 'output_B': 'categorical_crossentropy'},
    loss_weights={'output_A': 0.5, 'output_B': 1.0}
)
```

### Training
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler]
)
```

### Callbacks
- **EarlyStopping**: Stop when metric stops improving
- **ReduceLROnPlateau**: Reduce LR when metric plateaus
- **ModelCheckpoint**: Save best model
- **TensorBoard**: Visualize training

---

## Debugging Checklist

### Model not learning (loss not decreasing)
- ✓ Check learning rate (try 1e-3, 1e-4)
- ✓ Verify data preprocessing (normalization)
- ✓ Check labels match outputs (shapes, ranges)
- ✓ Try simpler model first
- ✓ Verify loss function matches problem type

### Overfitting (train >> val performance)
- ✓ Add dropout
- ✓ Add L2 regularization
- ✓ Reduce model capacity
- ✓ Get more training data
- ✓ Use data augmentation
- ✓ Early stopping

### Underfitting (both train and val poor)
- ✓ Increase model capacity
- ✓ Train longer
- ✓ Reduce regularization
- ✓ Check data quality
- ✓ Verify feature engineering

### Memory issues
- ✓ Reduce batch size
- ✓ Reduce model size
- ✓ Use mixed precision training
- ✓ Use gradient accumulation

---

## Quick Reference: Common Shapes

```
Dense layer:           (batch, input_dim) → (batch, units)
Conv2D:                (batch, h, w, c) → (batch, h', w', filters)
MaxPooling2D:          (batch, h, w, c) → (batch, h/2, w/2, c)
Flatten:               (batch, h, w, c) → (batch, h*w*c)
GlobalAvgPool2D:       (batch, h, w, c) → (batch, c)
Embedding:             (batch, seq_len) → (batch, seq_len, embed_dim)
LSTM/GRU:              (batch, steps, feats) → (batch, units) or (batch, steps, units)
```

---

## Common Pitfalls

1. **Wrong loss function**: Binary cross-entropy for multi-class (use categorical)
2. **Wrong final activation**: Softmax for regression (use linear)
3. **Not normalizing inputs**: Raw pixel values [0, 255] instead of [0, 1]
4. **Data leakage**: Using test data for preprocessing (fit only on train)
5. **Wrong input shape**: Missing channel dimension for images
6. **Wrong label format**: One-hot when using sparse_categorical_crossentropy
7. **Vanishing/exploding gradients**: Use batch norm, gradient clipping, or better initialization
8. **Learning rate too high**: Loss oscillates or diverges
9. **Not shuffling training data**: Model learns order instead of patterns

---

**Reference**: Based on _Deep Learning with Python_ (2nd Edition) by François Chollet  
**Course**: COSC3007 - Deep Learning, RMIT University
