# ASL (American Sign Language) Recognition System - Architecture

A complete breakdown of how the entire ASL recognition program works, from data preparation to real-time inference.

---

## System Overview

```mermaid
flowchart TB
    subgraph DataPrep["📦 Data Preparation Pipeline"]
        direction TB
        RawData["Raw ASL Images<br/>(A-Z folders)"]
        DC["data_compression.py<br/>balance_compress_npz()"]
        DA["data_augmentation.py<br/>augment_image()"]
        NPZ["Compressed .npz Dataset<br/>(images + labels)"]

        RawData --> DC
        DC --> DA
        DA --> DC
        DC --> NPZ
    end

    subgraph Training["🎯 Training Pipeline"]
        direction TB
        TE["train_evaluate.py<br/>train_evaluate()"]
        DS["asl_dataset.py<br/>ASLDataset"]
        LM["label_mapper.py<br/>LabelMapper"]
        TM["train_model.py<br/>train_model()"]
        RE["run_epoch()"]

        NPZ --> TE
        TE --> DS
        DS --> LM
        TE --> TM
        TM --> RE
    end

    subgraph Models["🧠 CNN Models (model/cnn_models.py)"]
        direction TB
        CNN["CustomCNN"]
        MN["CustomMobileNetV2<br/>(Primary)"]
        RN["CustomResNet18"]
        CA["ChannelAttention<br/>(attention_layers.py)"]

        CNN --> CA
        MN --> CA
        RN --> CA
    end

    subgraph Evaluation["📊 Evaluation"]
        direction TB
        EM["evaluate_model.py<br/>evaluate_model()"]
        ET["evaluation_tools.py"]
        PP["plot_performance_history()"]
        CM["print_confusion_matrix()"]

        EM --> ET
        ET --> PP
        ET --> CM
    end

    subgraph App["🖐️ Real-Time Application (app/)"]
        direction TB
        FR["frame.py<br/>HandDetectionApp"]
        FU["frame_utils.py"]
        MP["MediaPipe Hands"]
        WC["Webcam Capture"]
        TK["Tkinter GUI"]

        FR --> FU
        FR --> MP
        FR --> WC
        FR --> TK
    end

    subgraph Utils["🔧 Utilities"]
        direction TB
        MC["model_checkpoint.py"]
        SM["save_checkpoint()"]
        LMD["load_model()"]

        MC --> SM
        MC --> LMD
    end

    Training --> Models
    Models --> Evaluation
    Training --> Evaluation
    Models --> Utils
    Utils --> App
```

---

## Detailed Component Breakdown

### 1. Data Preparation Pipeline

```mermaid
flowchart LR
    subgraph Input
        RAW["Raw Image Folders<br/>/A, /B, /C... /Z"]
    end

    subgraph Balance["Class Balancing"]
        GCC["get_class_counts()<br/>Count images per class"]
        MAX["Find max_count<br/>(largest class)"]
        NEED["Calculate num_images_needed<br/>max_count - current + base_aug"]
    end

    subgraph Augmentation["Data Augmentation"]
        AUG["augment_image()"]
        ROT["Random Rotation<br/>(-15° to 15°)"]
        FLIP["Horizontal Flip<br/>(50% chance)"]
        BRIGHT["Brightness Adjust<br/>(0.8x to 1.2x)"]
        SCALE["Random Scale<br/>(0.9x to 1.1x)"]
        TRANS["Random Translation<br/>(±10% shift)"]
    end

    subgraph Output
        SHUFFLE["sklearn.shuffle()"]
        COMPRESS["np.savez_compressed()"]
        NPZ["compressed_asl_crop_v4.npz<br/>images[] + labels[]"]
    end

    RAW --> GCC --> MAX --> NEED
    NEED --> AUG
    AUG --> ROT & FLIP & BRIGHT & SCALE & TRANS
    ROT & FLIP & BRIGHT & SCALE & TRANS --> SHUFFLE --> COMPRESS --> NPZ
```

---

### 2. Model Architecture

```mermaid
flowchart TB
    subgraph MobileNetV2["CustomMobileNetV2 (Primary Model)"]
        direction TB
        BASE["MobileNetV2 Base<br/>(Pretrained ImageNet)"]

        subgraph Features["Feature Extraction"]
            F0["features[0]<br/>32 channels"]
            F1["features[1-N]"]
            ATT48["ChannelAttention(48)"]
            ATT96["ChannelAttention(96)"]
            ATT128["ChannelAttention(128)"]
            ATT256["ChannelAttention(256)"]
            ATT512["ChannelAttention(512)"]
        end

        subgraph Classifier["Custom Classifier"]
            DROP1["Dropout(0.2)"]
            FC1["Linear(last_channel → 512)"]
            RELU["ReLU"]
            DROP2["Dropout(0.5)"]
            FC2["Linear(512 → 26 classes)"]
        end

        GAP["Global Average Pooling<br/>x.mean([2, 3])"]

        BASE --> F0 --> F1
        F1 --> ATT48 --> ATT96 --> ATT128 --> ATT256 --> ATT512
        ATT512 --> GAP --> DROP1 --> FC1 --> RELU --> DROP2 --> FC2
    end

    subgraph Attention["Channel Attention Module"]
        direction LR
        AVG["AdaptiveAvgPool2d(1)"]
        FC_A["Conv2d(in → in/8)"]
        RELU_A["ReLU"]
        FC_B["Conv2d(in/8 → in)"]
        SIG["Sigmoid"]
        MUL["x × attention"]

        AVG --> FC_A --> RELU_A --> FC_B --> SIG --> MUL
    end

    INPUT["Input Image<br/>(3, 224, 224)"] --> MobileNetV2
    MobileNetV2 --> OUTPUT["Output<br/>26 class probabilities<br/>(A-Z)"]
```

---

### 3. Training Flow

```mermaid
sequenceDiagram
    participant Main as train_evaluate.py
    participant Data as ASLDataset
    participant Loader as DataLoader
    participant Model as CustomMobileNetV2
    participant Train as train_model()
    participant Epoch as run_epoch()
    participant GPU as CUDA Device

    Main->>Main: decompress_npz()
    Main->>Main: train_test_split(80/20)
    Main->>Data: Create train/val datasets
    Data->>Data: LabelMapper.label_to_index()
    Main->>Loader: batch_size=100, shuffle=True
    Main->>Model: Initialize with num_classes=26
    Main->>GPU: model.to(device)

    loop For each epoch (1 to num_epochs)
        Main->>Train: train_model()
        Train->>Epoch: run_epoch(train=True)

        loop For each batch
            Epoch->>GPU: images, labels to device
            Epoch->>Model: Forward pass
            Model-->>Epoch: outputs
            Epoch->>Epoch: CrossEntropyLoss
            Epoch->>Model: Backward pass
            Epoch->>Model: optimizer.step()
        end

        Train->>Epoch: run_epoch(train=False)
        Note over Epoch: Validation (no gradients)

        alt Every 5 epochs
            Train->>Train: Save checkpoint
        end
    end

    Main->>Main: plot_performance_history()
    Main->>Main: evaluate_model()
```

---

### 4. Real-Time Inference Application

```mermaid
flowchart TB
    subgraph GUI["Tkinter GUI (HandDetectionApp)"]
        direction TB
        TITLE["Title: Hand Sign Detection"]
        CANVAS["Video Canvas<br/>(640x480)"]

        subgraph Buttons
            START["Start Button"]
            STOP["Stop Button"]
            EXIT["Exit Button"]
            DEL["Delete Button"]
            SPACE["Space Button"]
            RESET["Reset Button"]
        end

        SENTENCE["Sentence Display Frame"]
    end

    subgraph Init["Initialization"]
        direction TB
        CAP["cv2.VideoCapture(0)"]
        MP_INIT["MediaPipe Hands<br/>detection_conf=0.7<br/>tracking_conf=0.7"]
        MODEL_LOAD["load_model()<br/>asl_crop_v4_1_mobilenet_weights.pth"]
        DEVICE["torch.device<br/>(cuda/cpu)"]
        QUEUE["predictions_queue<br/>deque(maxlen=10)"]
    end

    subgraph Frame["Frame Processing Loop"]
        direction TB
        READ["cap.read()"]
        PROCESS["process_frame()"]
        CONVERT["cv2 → PIL → ImageTk"]
        DISPLAY["canvas.create_image()"]
        SCHEDULE["window.after(10ms)"]
    end

    Init --> GUI
    START --> Frame
    READ --> PROCESS --> CONVERT --> DISPLAY --> SCHEDULE --> READ
```

---

### 5. Frame Processing Pipeline

```mermaid
flowchart TB
    subgraph Input
        FRAME["Raw Webcam Frame<br/>(BGR)"]
    end

    subgraph MediaPipe["MediaPipe Hand Detection"]
        RGB["Convert BGR → RGB"]
        DETECT["hands.process(rgb_frame)"]
        LANDMARKS["Extract 21 Hand Landmarks"]
    end

    subgraph Visualization
        BBOX["draw_bounding_box()<br/>Green box around hand"]
        PALM["draw_palm_connections()<br/>Gray lines"]
        HAND["draw_hand_landmarks()<br/>Colored circles by finger"]
    end

    subgraph Extraction
        CROP["get_hand_bounding_box()<br/>Crop hand region"]
        BLACK["Create black background"]
        FEATURES["Extract hand features"]
    end

    subgraph Prediction["Sign Prediction"]
        TRANSFORM["Image Transform<br/>Resize(224,224)<br/>ToTensor()<br/>Normalize()"]
        INFER["model(image)"]
        SOFTMAX["torch.softmax()"]
        CONF["Check confidence > 0.7"]
        LABEL["LabelMapper.index_to_label()"]
    end

    subgraph Smoothing["Prediction Smoothing"]
        QUEUE2["predictions_queue<br/>(last 10 frames)"]
        MOST["Most frequent prediction"]
        DELAY["2 second delay<br/>between outputs"]
    end

    subgraph Output
        ADD["add_to_sentence()"]
        DISPLAY2["Update sentence_label"]
    end

    FRAME --> RGB --> DETECT --> LANDMARKS
    LANDMARKS --> BBOX & PALM & HAND
    LANDMARKS --> CROP --> BLACK --> FEATURES
    FEATURES --> TRANSFORM --> INFER --> SOFTMAX --> CONF
    CONF -->|Yes| LABEL --> QUEUE2 --> MOST --> DELAY --> ADD --> DISPLAY2
    CONF -->|No| SKIP["Skip prediction"]
```

---

### 6. Label Mapping System

```mermaid
classDiagram
    class LabelMapper {
        +list labels = ["A", "B", ..., "Z"]
        +dict label_to_index_map
        +dict index_to_label_map
        +label_to_index(label) int
        +index_to_label(index) str
    }

    class ASLDataset {
        +list images
        +list labels
        +callable transform
        +ndarray numerical_labels
        +__len__() int
        +__getitem__(idx) tuple
    }

    LabelMapper <-- ASLDataset : uses
```

| Index | Label | Index | Label | Index | Label |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 0     | A     | 10    | K     | 20    | U     |
| 1     | B     | 11    | L     | 21    | V     |
| 2     | C     | 12    | M     | 22    | W     |
| 3     | D     | 13    | N     | 23    | X     |
| 4     | E     | 14    | O     | 24    | Y     |
| 5     | F     | 15    | P     | 25    | Z     |
| 6     | G     | 16    | Q     |       |       |
| 7     | H     | 17    | R     |       |       |
| 8     | I     | 18    | S     |       |       |
| 9     | J     | 19    | T     |       |       |

---

### 7. File Structure Overview

```mermaid
graph TB
    subgraph Root["Sign-Language-Recognition/"]
        subgraph App["app/"]
            F1["frame.py<br/>Main GUI Application"]
            F2["frame_utils.py<br/>Frame Processing Functions"]
        end

        subgraph Model["model/"]
            M1["cnn_models.py<br/>CustomCNN, MobileNetV2, ResNet18"]
            M2["attention_layers.py<br/>ChannelAttention"]
            M3["asl_dataset.py<br/>PyTorch Dataset Class"]
        end

        subgraph Train["train/"]
            T1["train_model.py<br/>run_epoch(), train_model()"]
            T2["train_evaluate.py<br/>Main Training Entry"]
        end

        subgraph Test["test/"]
            TE1["evaluate_model.py<br/>Model Evaluation"]
        end

        subgraph Compressor["compressor/"]
            C1["data_compression.py<br/>NPZ Compression/Decompression"]
            C2["data_augmentation.py<br/>Image Augmentation"]
        end

        subgraph Utils["utils/"]
            U1["label_mapper.py<br/>A-Z Label Encoding"]
            U2["evaluation_tools.py<br/>Plots & Confusion Matrix"]
            U3["model_checkpoint.py<br/>Save/Load Models"]
        end

        subgraph Data["data/weights/"]
            W1["asl_crop_v4_1_mobilenet_weights.pth"]
        end
    end
```

---

## Complete End-to-End Flow

```mermaid
flowchart LR
    subgraph Phase1["1. Data Collection"]
        A1["Collect ASL<br/>hand images"]
        A2["Organize into<br/>A-Z folders"]
    end

    subgraph Phase2["2. Preprocessing"]
        B1["Balance classes"]
        B2["Augment images"]
        B3["Compress to .npz"]
    end

    subgraph Phase3["3. Training"]
        C1["Load dataset"]
        C2["Initialize MobileNetV2"]
        C3["Train with Adam"]
        C4["Save best weights"]
    end

    subgraph Phase4["4. Evaluation"]
        D1["Load test set"]
        D2["Run inference"]
        D3["Generate metrics"]
    end

    subgraph Phase5["5. Deployment"]
        E1["Load trained model"]
        E2["Start webcam"]
        E3["Detect hands"]
        E4["Classify signs"]
        E5["Display sentence"]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4 --> Phase5
```

---

## Key Parameters

| Component        | Parameter                | Value   | Description               |
| ---------------- | ------------------------ | ------- | ------------------------- |
| **MediaPipe**    | min_detection_confidence | 0.7     | Hand detection threshold  |
| **MediaPipe**    | min_tracking_confidence  | 0.7     | Hand tracking threshold   |
| **Model**        | Input Size               | 224×224 | Image dimensions          |
| **Model**        | Num Classes              | 26      | A-Z letters               |
| **Training**     | Batch Size               | 100     | Samples per batch         |
| **Training**     | Learning Rate            | 0.001   | Adam optimizer            |
| **Training**     | Epochs                   | 10      | Training iterations       |
| **Inference**    | Confidence Threshold     | 0.7     | Min prediction confidence |
| **Inference**    | Prediction Window        | 10      | Frames to average         |
| **Inference**    | Prediction Delay         | 2s      | Cooldown between outputs  |
| **Augmentation** | Base Aug                 | 500     | Extra samples per class   |

---

## Technologies Used

- **PyTorch** - Deep learning framework
- **MediaPipe** - Hand landmark detection
- **OpenCV** - Video capture and image processing
- **Tkinter** - Desktop GUI framework
- **TorchVision** - MobileNetV2/ResNet18 pretrained models
- **NumPy** - Array operations and .npz compression
- **scikit-learn** - Train/test split and metrics
- **Matplotlib/Seaborn** - Visualization and evaluation plots
