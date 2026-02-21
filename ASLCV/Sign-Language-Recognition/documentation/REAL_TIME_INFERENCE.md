# Real-Time ASL Inference Application - Deep Dive

A comprehensive, detailed breakdown of the real-time American Sign Language recognition application.

---

## Table of Contents

1. [Application Overview](#application-overview)
2. [Entry Point & Initialization](#entry-point--initialization)
3. [GUI Architecture](#gui-architecture)
4. [Video Capture Loop](#video-capture-loop)
5. [MediaPipe Hand Detection](#mediapipe-hand-detection)
6. [Hand Visualization System](#hand-visualization-system)
7. [Prediction Pipeline](#prediction-pipeline)
8. [Prediction Smoothing System](#prediction-smoothing-system)
9. [Sentence Building](#sentence-building)
10. [Complete Data Flow](#complete-data-flow)

---

## Application Overview

The real-time inference application is a desktop GUI that captures webcam video, detects hands using MediaPipe, extracts hand landmarks, and uses a trained MobileNetV2 model to classify ASL letters (A-Z) in real-time.

```mermaid
flowchart TB
    subgraph Files["Source Files"]
        F1["frame.py<br/>(187 lines)"]
        F2["frame_utils.py<br/>(224 lines)"]
    end

    subgraph Classes["Main Components"]
        C1["HandDetectionApp<br/>Main application class"]
        C2["MediaPipe Hands<br/>Hand landmark detection"]
        C3["CustomMobileNetV2<br/>Sign classification"]
        C4["Tkinter GUI<br/>User interface"]
    end

    F1 --> C1
    F2 --> C1
    C1 --> C2 & C3 & C4
```

---

## Entry Point & Initialization

### Application Startup

```python
# frame.py - Lines 183-186
if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root, "Hand Detection with Mediapipe")
```

### Initialization Sequence

```mermaid
sequenceDiagram
    participant Main as __main__
    participant TK as Tkinter
    participant App as HandDetectionApp
    participant MP as MediaPipe
    participant Model as PyTorch Model
    participant Cam as Webcam

    Main->>TK: tk.Tk()
    Main->>App: HandDetectionApp(root, title)

    Note over App: __init__ begins

    App->>App: Configure window (900x650)
    App->>MP: Initialize Hands model
    Note over MP: detection_conf=0.7<br/>tracking_conf=0.7

    App->>App: Set constants
    Note over App: PREDICTION_WINDOW=10<br/>PREDICTION_DELAY=2s<br/>CONFIDENCE_THRESHOLD=0.7

    App->>App: Detect device (CUDA/CPU)
    App->>Model: load_model(weights_path)
    App->>Model: model.eval()

    App->>App: Setup transforms
    App->>Cam: VideoCapture(0)

    alt Webcam Opens
        App->>App: Build GUI components
    else Webcam Fails
        App->>Main: exit()
    end

    App->>TK: mainloop()
```

### Initialization Constants

| Constant               | Value     | Purpose                                         |
| ---------------------- | --------- | ----------------------------------------------- |
| `PREDICTION_WINDOW`    | 10        | Number of frames to average predictions over    |
| `PREDICTION_DELAY`     | 2 seconds | Minimum time between adding letters to sentence |
| `CONFIDENCE_THRESHOLD` | 0.7       | Minimum confidence to accept a prediction       |
| `Window Size`          | 900×650   | Tkinter window dimensions                       |
| `Canvas Size`          | 640×480   | Video display dimensions                        |

### Device Detection

```mermaid
flowchart LR
    CHECK["torch.cuda.is_available()"]

    CHECK -->|True| CUDA["device = 'cuda'<br/>GPU Acceleration"]
    CHECK -->|False| CPU["device = 'cpu'<br/>CPU Processing"]

    CUDA --> LOAD["Load model to device"]
    CPU --> LOAD
```

---

## GUI Architecture

### Component Layout

```mermaid
flowchart TB
    subgraph Window["Tkinter Window (900x650)"]
        direction TB

        TITLE["Title Label<br/>'Hand Sign Detection'<br/>Font: Arial 24 Bold<br/>Color: #FFFFFF on #333333"]

        subgraph VideoFrame["Video Frame"]
            CANVAS["Canvas<br/>640x480 pixels<br/>bg: #222222<br/>Sunken relief"]
        end

        subgraph ButtonFrame["Button Frame (2 rows)"]
            direction LR
            subgraph Row1["Row 1"]
                B_START["Start<br/>#4CAF50"]
                B_STOP["Stop<br/>#f44336"]
                B_EXIT["Exit<br/>#555555"]
            end
            subgraph Row2["Row 2"]
                B_DEL["Delete<br/>#FF4081"]
                B_SPACE["Space<br/>#00BCD4"]
                B_RESET["Reset<br/>#9C27B0"]
            end
        end

        subgraph SentenceFrame["Sentence Frame"]
            SENTENCE["Sentence Label<br/>Font: Arial 18<br/>wraplength: 800px"]
        end

        TITLE --> VideoFrame --> ButtonFrame --> SentenceFrame
    end
```

### Button Functions

```mermaid
flowchart LR
    subgraph Controls
        START["▶ Start"] -->|"start_video()"| RUN["running = True<br/>update_frame()"]
        STOP["⏹ Stop"] -->|"stop_video()"| HALT["running = False"]
        EXIT["✕ Exit"] -->|"window.quit()"| CLOSE["Close application"]
    end

    subgraph TextEditing
        DELETE["Delete"] -->|"delete_last_sign()"| DEL["sentence = sentence[:-1]"]
        SPACE["Space"] -->|"add_space()"| SPC["sentence += ' '"]
        RESET["Reset"] -->|"reset_sentence()"| CLR["sentence = ''"]
    end
```

### Color Scheme

| Component      | Background | Foreground | Hex Code              |
| -------------- | ---------- | ---------- | --------------------- |
| Window         | Dark       | -          | `#333333`             |
| Canvas         | Darker     | -          | `#222222`             |
| Title          | Dark       | White      | `#333333` / `#FFFFFF` |
| Start Button   | Green      | White      | `#4CAF50`             |
| Stop Button    | Red        | White      | `#f44336`             |
| Exit Button    | Gray       | White      | `#555555`             |
| Delete Button  | Pink       | White      | `#FF4081`             |
| Space Button   | Cyan       | White      | `#00BCD4`             |
| Reset Button   | Purple     | White      | `#9C27B0`             |
| Sentence Frame | Gray       | -          | `#555555`             |

---

## Video Capture Loop

### Main Loop Flow

```mermaid
flowchart TB
    START["start_video()"] --> CHECK_RUN{"self.running?"}

    CHECK_RUN -->|No| END["Stop"]
    CHECK_RUN -->|Yes| UPDATE["update_frame()"]

    UPDATE --> READ["ret, frame = cap.read()"]

    READ --> CHECK_RET{"ret == True?"}
    CHECK_RET -->|No| ERROR["Print error"]
    CHECK_RET -->|Yes| PROCESS["process_frame(self, frame)"]

    PROCESS --> CONVERT["Convert BGR→RGB<br/>frame → PIL Image"]
    CONVERT --> PHOTO["ImageTk.PhotoImage()"]
    PHOTO --> DISPLAY["canvas.create_image()"]

    DISPLAY --> SCHEDULE["window.after(10ms, update_frame)"]
    SCHEDULE --> CHECK_RUN

    ERROR --> SCHEDULE
```

### Frame Rate Calculation

The loop runs with a 10ms delay between frames:

- **Theoretical max**: 100 FPS
- **Actual rate**: ~30-60 FPS (limited by webcam and processing time)

```python
# frame.py - Lines 127-146
def update_frame(self):
    if self.running:
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Could not read frame.")
            pass

        # Process the frame (hand detection + prediction)
        process_frame(self, frame)

        # Convert BGR (OpenCV) to RGB (PIL)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.imgtk = ImageTk.PhotoImage(image=img)

        # Display on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        # Schedule next frame (10ms = ~100 FPS max)
        self.window.after(10, self.update_frame)
```

---

## MediaPipe Hand Detection

### MediaPipe Configuration

```mermaid
flowchart LR
    subgraph Config["MediaPipe Hands Configuration"]
        DETECT["min_detection_confidence: 0.7"]
        TRACK["min_tracking_confidence: 0.7"]
    end

    subgraph Output["21 Hand Landmarks"]
        direction TB
        L0["0: WRIST"]
        L1["1-4: THUMB"]
        L5["5-8: INDEX"]
        L9["9-12: MIDDLE"]
        L13["13-16: RING"]
        L17["17-20: PINKY"]
    end

    Config --> Output
```

### 21 Hand Landmarks Diagram

```
                    8 (Index Tip)
                    |
                    7
                    |
                    6
                   /
    4 (Thumb)     5
    |            /
    3           /
    |          /
    2    12   9    16   20
    |     |    \    |    |
    1    11        15   19
     \    |    \    |    |
      \  10   13---14   18
       \  |    |    |    |
        \ |    |    |   /
         0----5----9---13---17
               (WRIST)
```

### Landmark Color Groups

```mermaid
flowchart TB
    subgraph ColorMap["Landmark Color Mapping"]
        direction LR

        subgraph Red["🔴 Red (Palm Base)"]
            R["0, 1, 5, 9, 13, 17"]
        end

        subgraph Orange["🟠 Orange (Thumb)"]
            O["2, 3, 4"]
        end

        subgraph Purple["🟣 Purple (Index)"]
            P["6, 7, 8"]
        end

        subgraph Yellow["🟡 Yellow (Middle)"]
            Y["10, 11, 12"]
        end

        subgraph Green["🟢 Green (Ring)"]
            G["14, 15, 16"]
        end

        subgraph Blue["🔵 Blue (Pinky)"]
            B["18, 19, 20"]
        end
    end
```

### BGR Color Definitions

```python
# frame_utils.py - Lines 14-22
colors = {
    'red': (45, 46, 255),       # Landmarks 0, 1, 5, 9, 13, 17
    'orange': (184, 231, 255),  # Landmarks 2, 3, 4
    'purple': (130, 61, 134),   # Landmarks 6, 7, 8
    'yellow': (6, 206, 255),    # Landmarks 10, 11, 12
    'green': (46, 255, 49),     # Landmarks 14, 15, 16
    'blue': (190, 103, 23),     # Landmarks 18, 19, 20
    'palm': (128, 128, 128)     # Palm connections (gray)
}
```

---

## Hand Visualization System

### Visualization Functions

```mermaid
flowchart TB
    subgraph Functions["Visualization Functions"]
        F1["draw_hand_landmarks()<br/>Draw colored circles for each landmark"]
        F2["draw_palm_connections()<br/>Draw gray lines for palm"]
        F3["draw_hand_features()<br/>Red circles + green connections"]
        F4["extract_hand_features_mask()<br/>Draw on black background"]
        F5["draw_bounding_box()<br/>Blue box + prediction label"]
        F6["detect_hand_bounds()<br/>Calculate box & draw"]
    end

    F1 --> FRAME["Modified Frame"]
    F2 --> FRAME
    F3 --> FRAME
    F4 --> MASK["Black Mask"]
    F5 --> FRAME
    F6 --> F5
```

### Bounding Box Visualization

```mermaid
flowchart TB
    subgraph BoundingBox["Bounding Box Structure"]
        direction TB

        TEXT_BOX["Filled Blue Rectangle<br/>(Prediction Text: 'A')"]
        MAIN_BOX["Blue Border Rectangle<br/>(Around Hand)"]
        HAND["Hand Landmarks"]

        TEXT_BOX --> MAIN_BOX --> HAND
    end
```

```
    ┌──────────────────┐
    │    Prediction    │  ← Filled blue box (20px height)
    ├──────────────────┤
    │                  │
    │   Hand Image     │  ← Blue border (2px)
    │   + Landmarks    │
    │                  │
    └──────────────────┘
```

### Palm Connections

```python
# frame_utils.py - Lines 32-33
palm_connections = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
```

```mermaid
flowchart LR
    subgraph Palm["Palm Hexagon"]
        P0((0)) --- P1((1))
        P0 --- P5((5))
        P5 --- P9((9))
        P9 --- P13((13))
        P13 --- P17((17))
        P17 --- P0
    end
```

---

## Prediction Pipeline

### Complete Prediction Flow

```mermaid
flowchart TB
    subgraph Input
        FRAME["Raw Frame (BGR)"]
    end

    subgraph Detection
        RGB["Convert to RGB"]
        MP["MediaPipe: hands.process()"]
        CHECK{"Hands detected?"}
    end

    subgraph FeatureExtraction
        MASK["Create black background<br/>np.zeros_like(frame)"]
        DRAW["draw_hand_features()"]
        EXTRACT["extract_hand_features_mask()"]
        MIRROR["cv2.flip(mask, 1)"]
    end

    subgraph Transform
        T1["ToPILImage()"]
        T2["Resize(224, 224)"]
        T3["ToTensor()"]
        T4["Normalize(ImageNet)"]
        T5["unsqueeze(0)"]
        T6["to(device)"]
    end

    subgraph Inference
        MODEL["model(orig_input)<br/>model(mirror_input)"]
        MAX_OUT["torch.max(orig, mirror)"]
        CONF_PRED["torch.max(output)"]
        CONF_CHECK{"confidence > 0.7?"}
    end

    subgraph Output
        LABEL["LabelMapper.index_to_label()"]
        QUEUE["predictions_queue.append()"]
        SMOOTH["smooth_predictions()"]
        DISPLAY["Draw bounding box"]
    end

    FRAME --> RGB --> MP --> CHECK
    CHECK -->|No| RETURN_NONE["return None"]
    CHECK -->|Yes| MASK --> DRAW --> EXTRACT --> MIRROR

    EXTRACT --> T1 --> T2 --> T3 --> T4 --> T5 --> T6
    MIRROR --> T1

    T6 --> MODEL --> MAX_OUT --> CONF_PRED --> CONF_CHECK
    CONF_CHECK -->|No| RETURN_NONE
    CONF_CHECK -->|Yes| LABEL --> QUEUE --> SMOOTH --> DISPLAY
```

### Mirror Augmentation Strategy

The system processes both the original and horizontally mirrored hand to improve robustness:

```mermaid
flowchart LR
    subgraph Original
        O1["Hand Features Mask"]
        O2["Transform & Normalize"]
        O3["Model Inference"]
        O4["Output: orig_output"]
    end

    subgraph Mirror
        M1["cv2.flip(mask, 1)"]
        M2["Transform & Normalize"]
        M3["Model Inference"]
        M4["Output: mirror_output"]
    end

    O4 & M4 --> MAX["torch.max(orig, mirror)"]
    MAX --> FINAL["Final Prediction"]
```

This approach helps with:

- **Left vs Right hand variations**
- **Mirrored sign ambiguities**
- **Increased prediction confidence**

### Image Transform Pipeline

```mermaid
flowchart LR
    A["NumPy Array<br/>(H, W, 3)"]
    --> B["PIL Image"]
    --> C["Resized<br/>(224, 224)"]
    --> D["Tensor<br/>(3, 224, 224)"]
    --> E["Normalized<br/>ImageNet stats"]
    --> F["Batched<br/>(1, 3, 224, 224)"]
    --> G["GPU/CPU<br/>Tensor"]
```

**Normalization values (ImageNet):**

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

---

## Prediction Smoothing System

### Why Smoothing?

Raw frame-by-frame predictions can be noisy due to:

- Slight hand movements
- Lighting changes
- Camera noise
- Transitional hand positions

### Smoothing Algorithm

```mermaid
flowchart TB
    NEW["New prediction arrives"]
    --> ADD["predictions_queue.append(prediction)"]
    --> CHECK{"queue length == 10?"}

    CHECK -->|No| WAIT["Return None<br/>(keep accumulating)"]
    CHECK -->|Yes| FIND["Find most frequent prediction<br/>max(set(queue), key=queue.count)"]

    FIND --> CLEAR["Clear queue"]
    CLEAR --> RETURN["Return smoothed prediction"]
```

### Prediction Queue Visualization

```
Frame 1:  [A] ________________
Frame 2:  [A, A] ______________
Frame 3:  [A, A, B] ____________
Frame 4:  [A, A, B, A] __________
Frame 5:  [A, A, B, A, A] ________
Frame 6:  [A, A, B, A, A, A] ______
Frame 7:  [A, A, B, A, A, A, B] ____
Frame 8:  [A, A, B, A, A, A, B, A] __
Frame 9:  [A, A, B, A, A, A, B, A, A]
Frame 10: [A, A, B, A, A, A, B, A, A, A]

→ Most frequent: 'A' (8 occurrences)
→ Output: 'A'
→ Queue cleared for next batch
```

### Timing Control

```mermaid
sequenceDiagram
    participant Frame as Frame Processing
    participant Queue as Prediction Queue
    participant Time as Time Check
    participant Sentence as Sentence Builder

    Note over Frame: Prediction 'A' received
    Frame->>Queue: Add to queue
    Queue->>Queue: Check if len == 10
    Queue-->>Frame: Smoothed: 'A'

    Frame->>Time: current_time - last_prediction_time

    alt Delay passed (> 2s)
        Time->>Sentence: add_to_sentence('A')
        Time->>Time: Update last_prediction_time
    else Delay not passed
        Time-->>Frame: Skip (too soon)
    end
```

---

## Sentence Building

### Sentence State Management

```mermaid
stateDiagram-v2
    [*] --> Empty: App starts

    Empty --> HasChars: add_to_sentence(sign)
    HasChars --> HasChars: add_to_sentence(sign)
    HasChars --> HasSpace: add_space()
    HasSpace --> HasChars: add_to_sentence(sign)
    HasChars --> Shorter: delete_last_sign()
    Shorter --> HasChars: add_to_sentence(sign)
    Shorter --> Empty: delete until empty
    HasChars --> Empty: reset_sentence()
    HasSpace --> Empty: reset_sentence()
```

### Sentence Functions

```python
# frame.py - Sentence Management Methods

def add_to_sentence(self, sign):
    """Add the recognized sign to the sentence."""
    self.sentence += sign
    self.sentence_label.config(text=self.sentence)

def delete_last_sign(self):
    """Delete the last character from the sentence."""
    self.sentence = self.sentence[:-1]
    self.sentence_label.config(text=self.sentence)

def add_space(self):
    """Add a space to the sentence."""
    self.sentence += " "
    self.sentence_label.config(text=self.sentence)

def reset_sentence(self):
    """Clear the entire sentence."""
    self.sentence = ""
    self.sentence_label.config(text=self.sentence)
```

### Example Sentence Building

```
Time    Action              Sentence
─────   ─────────────────   ─────────────────
0.0s    App starts          ""
2.5s    Predict 'H'         "H"
5.0s    Predict 'E'         "HE"
7.5s    Predict 'L'         "HEL"
10.0s   Predict 'L'         "HELL"
12.5s   Predict 'O'         "HELLO"
-       Click [Space]       "HELLO "
15.0s   Predict 'W'         "HELLO W"
17.5s   Predict 'O'         "HELLO WO"
20.0s   Predict 'R'         "HELLO WOR"
22.5s   Predict 'L'         "HELLO WORL"
25.0s   Predict 'D'         "HELLO WORLD"
```

---

## Complete Data Flow

### End-to-End Pipeline

```mermaid
flowchart TB
    subgraph Hardware["1. Hardware Layer"]
        WEBCAM["🎥 Webcam<br/>(cv2.VideoCapture)"]
        GPU["🖥️ GPU/CPU<br/>(torch.device)"]
    end

    subgraph Capture["2. Frame Capture"]
        READ["cap.read()"]
        BGR["BGR Frame"]
    end

    subgraph Detection["3. Hand Detection"]
        RGB["BGR → RGB"]
        MEDIAPIPE["MediaPipe Hands"]
        LANDMARKS["21 Landmarks<br/>(x, y, z normalized)"]
    end

    subgraph Visualization["4. Visualization"]
        DRAW_LIVE["Draw on live frame:<br/>- Colored landmarks<br/>- Connections<br/>- Bounding box"]
        DRAW_MASK["Draw on black mask:<br/>- Palm connections<br/>- Colored landmarks"]
    end

    subgraph Prediction["5. Classification"]
        TRANSFORM["Image Transforms<br/>(Resize, Normalize)"]
        MODEL["MobileNetV2<br/>+ ChannelAttention"]
        SOFTMAX["26-class output<br/>(A-Z probabilities)"]
        CONF["Confidence check<br/>(> 0.7)"]
    end

    subgraph Smoothing["6. Smoothing"]
        QUEUE["Prediction Queue<br/>(10 frames)"]
        VOTE["Majority Vote"]
        DELAY["2s Delay Check"]
    end

    subgraph Output["7. Output"]
        SENTENCE["Sentence String"]
        LABEL["GUI Label"]
        DISPLAY["Canvas Display"]
    end

    WEBCAM --> READ --> BGR
    BGR --> RGB --> MEDIAPIPE --> LANDMARKS
    LANDMARKS --> DRAW_LIVE --> DISPLAY
    LANDMARKS --> DRAW_MASK --> TRANSFORM --> MODEL --> SOFTMAX --> CONF
    CONF --> QUEUE --> VOTE --> DELAY --> SENTENCE --> LABEL

    GPU -.-> MODEL
```

### Timing Analysis

| Stage               | Approximate Time | Notes                        |
| ------------------- | ---------------- | ---------------------------- |
| Frame capture       | ~10-30ms         | Depends on webcam            |
| MediaPipe detection | ~15-25ms         | GPU accelerated if available |
| Feature extraction  | ~5ms             | Drawing on mask              |
| Image transforms    | ~5ms             | Resize + normalize           |
| Model inference     | ~10-30ms         | GPU: ~10ms, CPU: ~30ms       |
| Smoothing           | ~0.1ms           | Simple counting              |
| GUI update          | ~5ms             | Tkinter refresh              |
| **Total per frame** | **~50-100ms**    | **10-20 FPS effective**      |

---

## Resource Management

### Cleanup on Exit

```python
# frame.py - Lines 177-180
def __del__(self):
    """Release resources when app is destroyed."""
    if self.cap.isOpened():
        self.cap.release()
```

### Memory Considerations

```mermaid
flowchart LR
    subgraph Persistent["Persistent in Memory"]
        MODEL["Model weights<br/>~25MB"]
        MEDIAPIPE["MediaPipe model<br/>~10MB"]
        QUEUE["Prediction queue<br/>(10 strings)"]
    end

    subgraph PerFrame["Per-Frame (garbage collected)"]
        FRAME["Frame array<br/>~900KB"]
        MASK["Feature mask<br/>~900KB"]
        TENSOR["Tensors<br/>~600KB"]
        IMGTK["ImageTk<br/>~900KB"]
    end
```

---

## Configuration Reference

### All Configurable Parameters

| Parameter                  | Location       | Default                                            | Description                             |
| -------------------------- | -------------- | -------------------------------------------------- | --------------------------------------- |
| `min_detection_confidence` | `frame.py:22`  | 0.7                                                | MediaPipe hand detection threshold      |
| `min_tracking_confidence`  | `frame.py:22`  | 0.7                                                | MediaPipe hand tracking threshold       |
| `PREDICTION_WINDOW`        | `frame.py:25`  | 10                                                 | Frames to average for smoothing         |
| `PREDICTION_DELAY`         | `frame.py:26`  | 2                                                  | Seconds between sentence updates        |
| `CONFIDENCE_THRESHOLD`     | `frame.py:27`  | 0.7                                                | Minimum confidence to accept prediction |
| `window.geometry`          | `frame.py:18`  | "900x650"                                          | Window dimensions                       |
| `canvas size`              | `frame.py:64`  | 640×480                                            | Video display size                      |
| `window.after`             | `frame.py:146` | 10                                                 | Milliseconds between frame updates      |
| `model_path`               | `frame.py:34`  | `data/weights/asl_crop_v4_1_mobilenet_weights.pth` | Trained weights file                    |

---

## Error Handling

### Current Error Checks

```mermaid
flowchart TB
    subgraph Checks["Error Handling Points"]
        C1["Webcam open check<br/>(exit if not opened)"]
        C2["Frame read check<br/>(print error, continue)"]
        C3["Hand detection<br/>(return None if no hands)"]
        C4["Confidence threshold<br/>(skip low confidence)"]
    end
```

### Potential Failure Points

| Scenario           | Current Handling   | Impact         |
| ------------------ | ------------------ | -------------- |
| Webcam not found   | `exit()`           | App terminates |
| Frame read fails   | Print + continue   | Skipped frame  |
| No hands detected  | Return None        | No prediction  |
| Low confidence     | Skip               | No prediction  |
| Model not found    | Uncaught exception | Crash          |
| CUDA out of memory | Uncaught exception | Crash          |

---

## Running the Application

```bash
# Activate your conda environment
conda activate asl-v_3

# Navigate to project root
cd Sign-Language-Recognition

# Run the application
python -m app.frame
```

### Expected Console Output

```
Using device: cuda
```

(or `cpu` if no GPU available)

---

## File Dependencies

```mermaid
flowchart TB
    subgraph App["app/"]
        FRAME["frame.py"]
        UTILS["frame_utils.py"]
    end

    subgraph Model["model/"]
        CNN["cnn_models.py"]
        ATT["attention_layers.py"]
    end

    subgraph Utils["utils/"]
        LM["label_mapper.py"]
        MC["model_checkpoint.py"]
    end

    subgraph Data["data/weights/"]
        WEIGHTS["asl_crop_v4_1_mobilenet_weights.pth"]
    end

    subgraph External["External Libraries"]
        TK["tkinter"]
        CV["opencv-python"]
        MP["mediapipe"]
        TORCH["pytorch"]
        PIL["pillow"]
    end

    FRAME --> UTILS
    FRAME --> LM
    FRAME --> MC
    MC --> CNN
    CNN --> ATT
    MC --> WEIGHTS

    FRAME --> TK & CV & MP & TORCH & PIL
    UTILS --> CV & MP
```
