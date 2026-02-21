## Project overview

Cerebro is a multimodal accessibility system designed to help disabled users and others interact with computers using alternative input methods. It enables full computer control via EEG, voice, computer vision (eye tracking), and sign language, with adaptive personalization through closed-loop error correction.

Cerebro uses an Electron-based frontend. Eye tracking controls the cursor in real time, allowing users to open and interact with an on-screen keyboard and type using only their gaze. Voice input enables command execution such as opening applications, typing text, or controlling games (e.g., "up", "down", "hit"). Sign language serves as an alternative input modality for users who are deaf, non-verbal, or unable to use voice.

EEG, eye cursor control, and sign language are fully functional automated runtime modalities whenever their toggles are enabled; none require operator handoffs during normal use.

Users can dynamically enable or disable individual input modalities and system features at runtime or in the middle of running it. Toggling a feature immediately starts or stops its associated processing pipeline, allowing users to customize the system to their specific needs without restarting the application. This makes Cerebro a fully modular and configurable accessibility platform.

The Electron frontend includes a real-time heads-up display and a small 3D visualizer (top-left) used during calibration and live control. It renders gaze rays / cursor target, EEG activity, active modalities, and the key metrics (latency, stability/jitter, and accuracy proxies) so users and judges can see what the system is doing and how it improves over time.

---

## Core system goals

### Accessibility control

Enable users to fully control a computer using gaze, voice, EEG, or sign language without requiring traditional mouse or keyboard input.

### Adaptive personalization

Continuously improve system accuracy and usability through per-user calibration, behavioral learning, and error correction.

### Real-time responsiveness

Maintain low-latency, real-time responsiveness suitable for cursor control, typing, and interactive applications.

### Real-time feasibility assumptions

- The strictest latency budget (50 ms) applies only to gaze-to-cursor control in the CV/eye-tracking path.
- EEG-to-intent and sign-language-to-intent control paths are baseline, fully functional paths with default budgets under 200 ms each.
- ASR EEG denoising and ErrP-related research pipelines are excluded from the 50 ms critical path.
- All baseline control outputs follow the canonical control/event contract defined below.
- All modality-specific pipelines are required to degrade gracefully when their measured latency exceeds budget rather than block input routing.
- Any pipeline latency spikes must be surfaced to the user and metrics pipeline in real time.

### Measurable improvement

Collect and expose performance metrics demonstrating accuracy, latency, stability, and improvement over time.

---

## Operating mode

### Baseline mode (default)

- EEG, voice, CV eye-tracking, and sign-language are fully functional automated control paths.
- No manual operator steps are required for normal operation.
- ASR/ErrP/experimental inference stacks are excluded from the baseline runtime contract.

### Research mode (opt-in)

- Enables optional EEG denoising, ErrP, and other research-only modules.
- Research modules must never reduce baseline control guarantees or override baseline safety contracts.
- Manual-in-loop features are permitted only for debugging or experiments and are not required for shipping functionality.

---

## Canonical control contracts (strict, required)

All control-capable messages MUST follow this envelope:

```json
{
  "source": "gaze|voice|eeg|sign|fusion|system",
  "timestamp": 1712345678,
  "confidence": 0.0,
  "intent": "mouse_move|mouse_click|mouse_hold|type_text|key_press|key_chord|key_down|key_up|toggle_mode|set_profile|noop|gaze_target",
  "payload": { ... }
}
```

Contract rules:

- `source`, `intent`, `timestamp`, and `confidence` are required for routing.
- `timestamp` is monotonic application time (ms) and MUST be comparable across agents.
- `confidence` is `0.0` to `1.0`; deterministic control-state updates may use `1.0`.
- `payload` must contain only fields valid for the declared intent.
- Invalid or malformed messages MUST be converted to `noop` with a reason.

Examples:

```json
{"source":"gaze","timestamp":1712345678,"confidence":1.0,"intent":"gaze_target","payload":{"target_x":842,"target_y":512}}
{"source":"voice","timestamp":1712345678,"confidence":0.92,"intent":"mouse_click","payload":{"button":"left"}}
{"source":"eeg","timestamp":1712345678,"confidence":0.78,"intent":"key_press","payload":{"keys":["SPACE"]}}
{"source":"sign","timestamp":1712345678,"confidence":0.95,"intent":"noop","payload":{"reason":"gesture_dropout"}}
```

### Cursor events

CV always emits:

```json
{"source":"gaze","timestamp":1712345678,"confidence":0.0,"intent":"gaze_target","payload":{"target_x":842,"target_y":512}}
```

### Discrete control events (voice/EEG/sign)

```json
{"source":"voice|eeg|sign","timestamp":1712345678,"confidence":0.0,"intent":"mouse_move","payload":{"dx":10,"dy":-5}}
{"source":"voice|eeg|sign","timestamp":1712345678,"confidence":0.0,"intent":"toggle_mode","payload":{"mode":"gaze_mouse"}}
{"source":"voice|eeg|sign","timestamp":1712345678,"confidence":0.0,"intent":"noop","payload":{"reason":"unrecognized_command"}}
```

### Fusion contract

- Input and output of the fusion agent MUST use canonical messages only.
- Fusion SHALL NOT invent unmodeled payload fields.

---

## Multimodal input agents

### EEG input agent

Handles EEG signal ingestion, preprocessing, feature extraction, and inference. Produces structured intent outputs compatible with control agents.

The EEG pipeline is fully automated in standard operation and runs continuously when enabled.

### EEG input data handling

- Capture EEG frames continuously from the headset stream and normalize to a single monotonic application clock.
- Apply session-level calibration constants from the user profile before feature extraction.
- Run lightweight artifact handling in-band for each frame window (bad-channel checks, baseline shift correction, amplitude clipping).
- Enforce deterministic timeout and retry logic; emit `noop` with reason on timeout.
- No operator-driven labeling, segmentation, or manual inference is required while control mode is active.

### EEG command contract (fully functional baseline)

```json
{"source":"eeg","timestamp":1712345678,"confidence":0.0,"intent":"mouse_move","payload":{"dx":<int>,"dy":<int>}}
{"source":"eeg","timestamp":1712345678,"confidence":0.0,"intent":"mouse_click","payload":{"button":"left"|"right"|"middle"}}
{"source":"eeg","timestamp":1712345678,"confidence":0.0,"intent":"mouse_hold","payload":{"button":"left"|"right"|"middle","state":"down"|"up"}}
{"source":"eeg","timestamp":1712345678,"confidence":0.0,"intent":"type_text","payload":{"text":"..."}}
{"source":"eeg","timestamp":1712345678,"confidence":0.0,"intent":"key_press","payload":{"keys":["CTRL","C"]}}
{"source":"eeg","timestamp":1712345678,"confidence":0.0,"intent":"noop","payload":{"reason":"..."}}
```

### EEG feasibility constraints

- The system supports baseline EEG control with consumer-grade channel sets using deterministic low-latency preprocessing.
- ErrP detection is an optional research mode and is never required for baseline control operation.
- When channel layout, quality, or sampling characteristics are below threshold, ErrP updates are disabled automatically while baseline EEG control continues with safe fallback outputs.
- Artifact cleaning for the real-time control path uses lightweight, low-latency filters and bad-channel checks; heavy denoisers like ASR that require multi-hundred-millisecond buffering are executed only for offline analysis or non-critical telemetry.
- For LSL-like EEG streams around 250 Hz, ASR-like pipelines are treated as non-real-time: documented end-to-end delay is approximately 1.092 seconds in typical settings, so they are excluded from deterministic control loops.

#### Recommended EEG processing split (Baseline / Research)

- Critical control path: lightweight baseline preprocessing, quick feature extraction, strict timeout budget, optional confidence filtering.
- Non-critical loop: optional ASR/advanced cleaning for post-hoc logging, dashboard quality analytics, and research experiments.

---

### Voice input agent

Handles voice recognition and command parsing.

### Backend (Gemini via OpenRouter)

- Use an OpenRouter-hosted Gemini model as the voice backend.
- Input: short microphone recordings (push-to-talk preferred) sent as base64-encoded audio.
- Output:
    - transcript text
    - structured command object for execution

### Recommended interaction mode

- Push-to-talk voice commands for highest reliability and lowest accidental activation risk.
- Chunk size: approximately 1–2 seconds per request.
- Avoid continuous streaming for the hackathon demo.

### Responsibilities

- Capture audio from the client.
- Send audio to Gemini via OpenRouter with a prompt enforcing strict JSON output.
- Validate returned JSON structure.
- Route validated commands to control agents.
- Provide safe fallback behavior:
    - Return `{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"noop","payload":{"reason":"..."}}` if parsing fails or confidence is low.
- If OpenRouter response latency exceeds a configured threshold:
  - discard the stale result,
  - emit `{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"noop","payload":{"reason":"latency exceeded"}}`,
  - raise a telemetry warning and await a new utterance.

### Command output contract (strict JSON)

The agent must always return exactly one canonical object in one of the following formats:

Mouse control:

```json
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"mouse_move","payload":{"dx":<int>,"dy":<int>}}
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"mouse_click","payload":{"button":"left"|"right"|"middle"}}
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"mouse_hold","payload":{"button":"left"|"right"|"middle","state":"down"|"up"}}
```

Keyboard control:

```json
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"type_text","payload":{"text":"..."}}
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"key_press","payload":{"keys":["CTRL","C"]}}
```

Mode and settings:

```json
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"toggle_mode","payload":{"mode":"gaze_mouse"|"voice_control"|"sign_control"}}
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"set_profile","payload":{"profile_id":"..."}}
```

Fallback:

```json
{"source":"voice","timestamp":1712345678,"confidence":0.0,"intent":"noop","payload":{"reason":"unrecognized_command"}}
```

### Safety and guardrails

- Only allow commands from an explicit allowlist.
- Require push-to-talk or wake phrase activation.
- Rate limit requests to prevent runaway loops or excessive API usage.

### Metrics to log (for demo and evaluation)

- Speech-to-text latency (ms)
- End-to-end command latency (ms)
- Command success rate (%)
- Parse failure rate (%)
- Repeat-needed count (per session and per user profile)

---

### Computer vision input agent

Handles eye tracking, gaze estimation, and gaze stabilization.

The CV path is a fully functioning real-time cursor modality that runs end-to-end from detection to fused control intent.

### Stabilization design (One Euro default, Kalman optional)

Webcam-based gaze tracking produces noisy measurements due to camera limitations, lighting variation, and natural eye micro-movements.

To ensure stable cursor control, the system applies a low-lag jitter-reduction layer before cursor control.

Primary stabilization policy:

- Primary filter is the One Euro filter (default), using speed-adaptive cutoff scheduling to suppress jitter at low velocity and preserve responsiveness at high velocity.
- Optional experimental filter: Kalman, disabled by default and only available in research mode.
- One Euro is used because it provides low latency, adaptive jitter suppression, minimal tuning burden, and better responsiveness during rapid shifts.
- Kalman may be enabled only when measured latency budgets remain satisfied and must not replace One Euro in primary cursor-control unless explicitly enabled as research mode.

Current implementation is MediaPipe-based, but the Computer vision input agent is model-agnostic and may be replaced with custom gaze models (including TAO or neural estimators) without changing downstream contracts.

The One Euro pipeline uses:

- gaze measurement coordinates from the active model backend (MediaPipe default)
- per-frame velocity estimate
- dynamically changing cutoff frequency based on motion speed

Pipeline integration:

`MediaPipe (or alternate model) → gaze estimation → screen mapping → One Euro → confidence-aware clamped target → cursor control agent`

Benefits:

- suppress high-frequency jitter
- preserve responsiveness for rapid gaze transitions
- maintain accuracy during detection noise
- improve usability and precision
- monitor and adapt to confidence drops to avoid lag spikes

#### Webcam robustness requirements

- Default design assumes consumer RGB webcam input and must tolerate changes in ambient light, screen reflections, and partial occlusion by adapting confidence thresholds and temporal smoothing.
- If repeated tracking dropout exceeds threshold, the cursor path is slowed and dwell activation is disabled until stable re-lock is achieved.
- IR/NIR camera support is a future enhancement for higher resilience; when unavailable, the filter and confidence gating settings must be tuned for stability-first behavior.

Runtime transport:

- The CV stack must publish stabilized targets via WebSocket or IPC channels to Electron, not file polling.

---

### Input fusion agent

Combines outputs from gaze, voice, EEG, and sign-language agents when multiple inputs are active. Resolves conflicts and selects the highest-confidence control signal.

### Fusion arbitration

- Every candidate event MUST conform to the canonical control contract.
- Priority order (default):
  1. Voice commands (explicit intent)
  2. EEG control intent (explicit)
  3. Sign-language intent (explicit)
  4. Gaze cursor control (continuous control)
- Explicit intent always overrides continuous control for a frame.
- If two explicit intents conflict:
  - choose highest confidence,
  - if equal, choose most recent timestamp,
  - if still tied, use user-selected source priority.
- Drop stale events older than 150 ms.
- Enforce per-intent cooldown windows for repeated identical commands.
- If the top candidate is invalid, unsafe, or below confidence threshold, emit `noop` and continue loop.
- No modality may indefinitely suppress all others; arbitration must output at most one executable event per 16 ms window or `noop`.

### Sign language input agent

Handles sign-language capture, gesture detection, and structured intent emission.

The sign-language path is fully automated in runtime use; manual intervention is limited to calibration/profile management only.

### Sign language command contract

```json
{"source":"sign","timestamp":1712345678,"confidence":0.0,"intent":"mouse_move","payload":{"dx":<int>,"dy":<int>}}
{"source":"sign","timestamp":1712345678,"confidence":0.0,"intent":"mouse_click","payload":{"button":"left"|"right"|"middle"}}
{"source":"sign","timestamp":1712345678,"confidence":0.0,"intent":"type_text","payload":{"text":"..."}}
{"source":"sign","timestamp":1712345678,"confidence":0.0,"intent":"key_press","payload":{"keys":["ENTER"]}}
{"source":"sign","timestamp":1712345678,"confidence":0.0,"intent":"noop","payload":{"reason":"..."}}
```

### Sign language responsibilities

- Run hand/gesture preprocessing and model inference as an automated runtime pipeline, not manual intervention.
- Use confidence + visibility gating; low-confidence outputs are converted to `noop` with reason.
- Emit explicit errors and fallback events when gesture input is not currently visible.
- Reuse global allowlist and safety checks before forwarding to keyboard/mouse control agents.

---

## Control and action agents

---

### Keyboard control agent

Translates parsed command outputs into OS-level keyboard events.

### Input contract

Receives keyboard intents from upstream agents:

- canonical event envelope (`source`, `timestamp`, `confidence`, `intent`, `payload`) as defined above.
- payload fields:
  - `text` (string, required for `type_text`)
  - `keys` (array of strings, required for key chords or hold/release)

Supported intents:

```json
{"intent":"type_text","payload":{"text":"..."}}
{"intent":"key_press","payload":{"keys":["ENTER"]}}
{"intent":"key_chord","payload":{"keys":["CTRL","C"]}}
{"intent":"key_down","payload":{"keys":["SHIFT"]}}
{"intent":"key_up","payload":{"keys":["SHIFT"]}}
{"intent":"noop","payload":{"reason":"..."}}
```

### Responsibilities

- Convert intents into OS-level keyboard events.
- Support typing, key presses, chords, and holds.
- Normalize key names across platforms.
- Ignore invalid or unsafe commands.

### Behavior requirements

- Low latency execution
- Deterministic output
- Idempotent key holds and releases

### Safety constraints

- Allowlist keys and intents only
- Block or confirm dangerous shortcuts
- Rate limit events to prevent runaway input

### Telemetry

Log:

- latency
- success rate
- blocked command count
- typing rate

---

### Cursor control agent

Translates gaze-derived control state into structured cursor-control messages.

### Input contract

Receives canonical gaze-target events:

- `source: "gaze"`
- `intent: "gaze_target"`
- `payload: {"target_x":<int>,"target_y":<int>}`
- `confidence` and `timestamp` per event contract

Cursor-control outputs are emitted only as canonical control events (`gaze_target` input -> `mouse_move`/`mouse_click`/`mouse_hold`/`noop` outputs) and should not rely on custom action fields.

### Responsibilities

- Convert stabilized gaze targets and action hints into structured cursor-control intents for the system interaction agent.
- Clamp targets and movement envelopes before emission.
- Emit click/hold/scroll intents, but never execute OS-level events directly.
- Support dwell-based click activation:
    - acquire dwell
    - show loader visualization
    - trigger click after dwell completes
- Execute dwell-based activation only when calibration state is complete.

### Behavior requirements

- Low latency cursor updates
- Stable motion without teleportation
- Deterministic behavior

### Safety constraints

- Allowlist permitted actions only
- Enforce screen bounds
- Rate limit click and hold execution
- If gaze confidence drops below active threshold, suppress dwell clicks until recovery criteria are met.

### Telemetry

Log:

- latency
- jitter
- overshoot rate
- click success rate
- rate limiting events

---

### System interaction agent

Interfaces with OS-level APIs for input control.

Responsibilities:

- Execute mouse and keyboard events
- Convert both relative (`dx`,`dy`) and absolute coordinates into OS-level cursor operations.
- Provide a secure abstraction layer between agents and the OS
- Enforce safety constraints and allowlists
- Support Windows, macOS, and Linux where possible

### Ownership boundary

- `cursor/keyboard control` emits structured commands only.
- `system interaction agent` is the only component that executes OS-level actions.

---

## Personalization and adaptation agents

### User profile agent

Stores and manages per-user calibration data, settings, and learned parameters.

### Responsibilities

- Create and maintain persistent user profiles.
- Store calibration parameters required for accurate gaze-to-screen mapping.
- Load profile automatically when the user opens the application.
- Save updates to calibration, correction parameters, and learned adjustments.
- Maintain separate profiles for different users and environments if needed.
- Own final persistence for all baseline/correction/learning parameter updates.

### Stored data (per user profile)

Calibration parameters:

- gaze-to-screen mapping coefficients
- head position normalization parameters
- One Euro filter and optional alternative filter tuning parameters (optional)

Correction parameters:

- offset corrections for systematic prediction error
- drift compensation values
- environment-specific corrections (camera position, lighting conditions)

Learned behavioral data:

- historical gaze accuracy metrics
- correction history
- stability metrics
- latency metrics

System settings:

- dwell timing settings
- cursor sensitivity settings
- input mode preferences

### Behavior requirements

- Automatically load the most recent user profile on application startup.
- Automatically save updated calibration and correction parameters.
- Ensure profile persistence across sessions.
- Never overwrite profile data without validation.

### Parameter ownership

- Calibration agent owns and writes baseline mapping state.
- Error correction agent owns temporary correction outputs for short-horizon stability.
- Learning loop agent owns persistent correction suggestions.
- User profile agent commits approved persistent updates from learning.

---

### Calibration agent

Handles initial gaze calibration, ongoing recalibration, and environment adaptation.

### Initial calibration process

When the user opens the application:

1. The calibration agent initiates calibration automatically.
2. The system displays calibration targets on screen.
3. The user is prompted to look at specific calibration points (center, left, right, top, bottom, or grid).
4. The system records gaze measurements for each calibration point.
5. The system computes mapping parameters from gaze space to screen space.
6. Calibration parameters are stored in the user profile.
7. The calibrated profile becomes active immediately.

This establishes the baseline gaze-to-screen transformation.

### Responsibilities

- Perform initial calibration for new users.
- Load existing calibration for returning users.
- Perform recalibration when drift or accuracy degradation is detected.
- Adapt calibration to changes in environment (camera movement, lighting, user posture).
- Produce baseline mapping and temporary correction seeds only.

### Calibration outputs

Produces:

- gaze-to-screen mapping coefficients
- baseline offset corrections
- initial eye-tracker smoothing state initialization (One Euro state/cache values)
- environment normalization parameters
- sign gesture primer calibration vector (dominant hand geometry, temporal signature)

These outputs are stored by the User Profile Agent.

### Behavior requirements

- Calibration must be deterministic and repeatable.
- Core gaze calibration must complete within a short time window suitable for demo use.
- Calibration should not require excessive user effort.
- Optional EEG/ErrP research calibration is isolated to opt-in research mode with explicit user permission and longer preparation time.

---

### Error correction agent

Analyzes prediction errors and applies corrective adjustments to maintain accuracy.

### Responsibilities

- Monitor gaze prediction vs expected target behavior.
- Detect systematic bias, drift, or offset errors.
- Apply correction parameters to compensate for prediction errors.
- Produce temporary correction outputs for runtime fusion/control.
- Do not use ErrP-only signals as the sole correction trigger.
- Use gaze/dwell/correction-event signals as primary, deterministic feedback for immediate adaptation.
- Accept ErrP feedback only when channel quality and confidence pass strict quality gates.

### Correction mechanisms

Error correction may include:

- screen-space offset correction
- gain adjustment
- drift compensation
- filtering and motion-model parameter tuning
- dynamic bias correction

### Error detection signals

Error correction may be triggered by:

- repeated user corrections
- dwell targeting errors
- cursor overshoot patterns
- increasing prediction variance
- explicit user feedback signals
- ErrP feedback that passes channel-coverage and confidence thresholds (optional research path)

### Behavior requirements

- Corrections must improve accuracy without destabilizing cursor behavior.
- Corrections must be applied gradually and safely.
- Corrections are proposals that require user profile persistence approval before commit.

### ErrP-specific constraints

- Frontocentral channel coverage (e.g., FCz/Cz/Fz-like signals) is required for reliable single-trial ErrP usage; low-density frontal-temporal layouts are considered insufficient.
- ErrP calibration must not block demo startup; the calibration flow uses a separate, optional “BCI research mode.”
- If ErrP classifier confidence is unstable or unknown, the agent must fall back to deterministic non-ErrP adaptation signals and emit a telemetry warning rather than attempting reinforcement updates.

---

### Learning loop agent

Continuously improves accuracy using user-specific performance feedback and historical data.

### Responsibilities

- Log gaze predictions, cursor targets, and correction outcomes.
- Analyze performance trends over time.
- Improve calibration and correction parameters based on usage data.
- Adapt to user-specific behavior patterns.

### Learning data collected

- gaze prediction accuracy
- cursor stabilization performance
- correction frequency
- dwell targeting success rate
- latency metrics
- prediction variance

### Continuous improvement process

1. System logs prediction and correction data during normal usage.
2. Learning agent analyzes trends and error patterns.
3. Learning agent updates correction parameters.
4. User profile agent stores approved persistent updates after policy checks.
5. Future predictions become more accurate.

### User feedback integration

User feedback mechanisms may include:

- explicit recalibration request
- dwell confirmation success or failure
- manual correction events
- optional explicit feedback actions (confirm accuracy / recalibrate)
- sign gesture correction events and visibility dropout recovery events

Feedback allows the system to:

- improve mapping accuracy
- reduce drift
- improve stability over time

### Behavior requirements

- Learning must improve accuracy over time.
- Learning must never introduce instability.
- Learning must operate safely and incrementally.
- Learning must persist across sessions through the user profile.

---

This now clearly defines the **full closed loop**:

- Calibration → baseline accuracy
- User Profile → persistence
- Error Correction → real-time fixes
- Learning Loop → long-term improvement

This is exactly what judges look for in accessibility + adaptive systems.

If you want next, I can also define the **actual data structure schema** for the user profile JSON so agents implement it correctly.

---

## Feedback and visualization agents

### Metrics collection agent

Collects, aggregates, and streams real-time system performance, prediction quality, and adaptation metrics from all active agents.

### Responsibilities

- Receive telemetry from all input, control, calibration, and learning agents.
- Record real-time performance data including:
    - gaze prediction coordinates
    - cursor position and stability
    - EEG signal features and inferred intent confidence
    - voice command latency and success rate
    - calibration accuracy and drift measurements
    - correction events and learning adjustments
- Compute derived performance metrics such as:
    - latency (input → system action)
    - jitter (variance in cursor movement)
    - stability score
    - prediction confidence trends
    - correction frequency
    - dwell success rate
- Maintain per-session and per-user metric histories.
- Stream structured telemetry to the Visualization agent and Demo metrics agent.

### Output contract (example)

```json
{
  "timestamp": 1712345678,
  "gaze": {
    "target_x": 842,
    "target_y": 512,
    "stability": 0.93,
    "jitter": 1.8
  },
  "cursor": {
    "latency_ms": 24,
    "overshoot_rate": 0.04
  },
  "voice": {
    "latency_ms": 310,
    "success_rate": 0.96
  },
  "system": {
    "active_modes": ["gaze", "voice"],
    "calibration_state": "complete"
  }
}
```

### Behavior requirements

- Must operate continuously during system runtime.
- Must introduce negligible latency.
- Must maintain accurate and consistent telemetry streams.

---

### Visualization agent

Provides real-time visualization of system state, sensor inputs, and prediction behavior through the Electron frontend and integrated 3D visualizer.

### Responsibilities

- Receive telemetry streams from the Metrics collection agent.
- Render live visualization elements including:
    - gaze position and cursor target
    - gaze stabilization behavior
- gaze vectors and predicted movement direction
- EEG signal activity and inferred intent indicators
- sign-language gesture state and recognition confidence
- active input modalities
- calibration targets and calibration state
- prediction confidence and stability indicators
- Drive the real-time 3D visualization panel in the Electron frontend.
- Provide visual feedback during:
    - calibration
    - cursor targeting
    - dwell activation
    - learning and correction events

### Visualization components

Electron frontend visualization includes:

- top-left 3D visualizer panel
- gaze and cursor overlay indicators
- calibration visualization markers
- active mode indicators
- stability and confidence visualization
- sign gesture confidence and visibility visualization

### Behavior requirements

- Must update in real time.
- Must accurately reflect system state.
- Must not block or slow control pipelines.
- Must remain responsive under continuous updates.

---

### Demo metrics agent

Provides structured, high-level performance metrics specifically designed for hackathon demonstration, evaluation, and measurable improvement validation.

### Responsibilities

- Aggregate key metrics into clear, demo-friendly indicators.
- Compute performance summaries including:
    - average system latency
    - gaze accuracy proxy metrics
    - cursor stability score
    - calibration improvement over time
    - correction frequency reduction
    - learning effectiveness indicators
- Expose metrics to the Electron frontend for display.
- Provide exportable metrics for demo presentation or logging.

### Demo metrics displayed in frontend

Examples:

- latency (ms)
- stability score
- accuracy proxy score
- calibration quality score
- improvement percentage over session
- active input modality status

### Example demo metrics output

```json
{
  "latency_ms": 26,
  "stability_score": 0.94,
  "accuracy_score": 0.91,
  "calibration_quality": 0.96,
  "improvement_percent": 18.4
}
```

### Behavior requirements

- Must clearly demonstrate measurable improvement over time.
- Must provide metrics understandable to judges and observers.
- Must operate continuously during system runtime.
- Must reflect real system performance accurately.

---

### Integration with Electron frontend

Data flow:

```
Input agents → Metrics collection agent → Visualization agent → Electron 3D visualizer
                                              ↓
                                       Demo metrics agent → Electron metrics display
```

---

## Frontend and visualization system

### Electron frontend agent

Provides the primary user interface, visualization, and interaction layer for Cerebro.

The frontend is implemented using Electron and serves as the real-time control panel and visualization interface for all system agents.

### Responsibilities

- Render the user interface for accessibility control.
- Display the on-screen keyboard controlled via gaze.
- Provide controls to enable and disable input modalities at runtime.
- Display system status, active agents, and current input mode.
- Provide visual feedback for dwell clicks, cursor targeting, and calibration.

### Real-time 3D visualization system

The Electron frontend includes a real-time 3D visualization panel located in the top-left region of the interface.

This visualizer displays live system state and sensor data, including:

- EEG signal activity and inferred brain-state features
- gaze direction and cursor targeting vectors
- stabilized gaze coordinates after adaptive low-pass filtering (One Euro default)
- calibration points and mapping accuracy
- input modality activity (gaze, voice, EEG, sign language)
- system confidence and prediction stability

### Visualization data inputs

The visualization system receives structured telemetry from:

- EEG input agent
- Computer vision input agent
- Calibration agent
- Error correction agent
- Metrics collection agent

Example visualization data contract:

```json
{
  "timestamp": 1712345678,
  "gaze": {
    "x": 842,
    "y": 512,
    "stability": 0.93
  },
  "eeg": {
    "signal_strength": 0.71,
    "active_channels": [3, 4, 7]
  },
  "sign": {
    "gesture": "left_click",
    "confidence": 0.94,
    "visible": true
  },
  "system": {
    "active_modes": ["gaze", "voice"],
    "calibration_state": "complete"
  }
}
```

### Visualization purposes

The visualization serves several critical roles:

- Provide real-time feedback to users
- Help users understand system behavior
- Support calibration and debugging
- Demonstrate adaptive learning and system improvement
- Provide transparent metrics for demo and evaluation

### Behavior requirements

- Visualization must update in real time.
- Visualization must not introduce control latency.
- Visualization must reflect true system state accurately.
- Visualization must remain responsive even during heavy processing.

### Integration with system architecture

```
Backend agents → telemetry → Electron frontend → visualization + UI
User input (UI toggles) → Electron frontend → agent activation/deactivation
```

---

## Data flow architecture

### Input pipeline

Sensors → preprocessing → inference → standardized intent events

WebSocket transport (runtime):

- CV and sensing pipelines publish control-ready events via WebSocket/IPC.
- File polling (including screen-position file writes) is prohibited for control routing.

Python CV transport path:

`Python CV process → gaze model inference → websocket/ipc → Electron`

### Control pipeline

Fusion/intent arbitration → cursor/keyboard agents → system interaction agent → OS

### Feedback pipeline

Performance metrics → calibration and learning agents → improved model parameters

---

## Model behavior guidelines

### Inference rules

Agents must produce structured, deterministic outputs.

- Structured outputs only: every agent output must follow its declared contract (JSON schema or message shape). No free-form text in control paths.
- Deterministic behavior: given the same inputs and configuration, the agent must produce the same outputs.
- Confidence and fallback: include confidence when available. If confidence is low or output fails validation, emit a noop intent with a reason.
- Non-blocking: inference must not block the real-time control loop. Heavy work must run asynchronously or off-thread.
- Bounded outputs: clamp and sanitize all numeric outputs such as screen coordinates, deltas, and rates.
- Mandatory temporal and confidence propagation: all control outputs must include monotonic `timestamp` and `confidence` and propagate through fusion, filtering, and control gating.
- Validation first: reject malformed payloads, unknown intents, or actions outside the allowlist.

---

### Adaptation rules

Adaptation must improve performance without introducing instability.

- Incremental updates only: apply small parameter adjustments over time, never large jumps.
- Safety bounds: enforce strict limits on offsets, gains, and filter tuning parameters.
- Reversible changes: maintain a last-known-good state and roll back if stability decreases.
- No direct control emissions: learning agents must not emit mouse or keyboard actions directly.
- Stability priority: if adaptation conflicts with stability, freeze adaptation and fall back to safe parameters.
- Per-user isolation: adaptation must remain scoped to individual user profiles.

---

### Calibration rules

Calibration must be repeatable, safe, and user-specific.

- Guided calibration: calibration should use clear visual targets and simple instructions.
- Repeatable calibration: similar conditions should produce similar calibration results.
- User-specific parameters: calibration outputs must be stored in the user profile.
- Environment awareness: detect camera movement, lighting changes, or posture changes and trigger recalibration or correction.
- Safe fallback: if calibration fails, use conservative default parameters and disable sensitive actions such as dwell click until stable.
- Traceable outputs: calibration must write explicit mapping parameters with timestamps.

---

## Metrics and evaluation requirements

### Required metrics

The system must collect and expose the following metrics per session and per user profile:

Accuracy metrics:

- dwell success rate
- calibration accuracy proxy
- correction frequency

Latency metrics:

- gaze pipeline latency
- voice pipeline latency
- EEG intent pipeline latency
- sign-language intent pipeline latency
- end-to-end system latency

Stability metrics:

- cursor jitter variance
- cursor overshoot rate
- detection dropout rate

Improvement metrics:

- stability improvement over time
- success rate improvement
- correction frequency reduction

---

### Demo metrics

The Electron frontend must display live performance metrics including:

- active input modalities
- calibration state
- current latency and average latency
- stability score
- dwell state indicator
- improvement trend indicators

These metrics must update continuously and reflect real system performance.

---

### Logging requirements

All agents must produce structured telemetry logs.

Logging requirements:

- Each record must include timestamp, agent name, event type, session ID, and profile ID.
- Logs must include latency, confidence, and relevant performance indicators.
- Logs must use structured formats such as JSON.
- Logging must not block real-time execution.
- Sensitive data such as raw audio or video must not be stored unless explicitly enabled.

---

## Visualization requirements

### Real-time visualization

The Electron frontend must render live system visualization including:

- gaze target and stabilized gaze position
- cursor position and motion trail
- calibration targets and calibration progress
- EEG signal summary indicators
- active input modalities
- dwell activation indicators
- sign gesture state, confidence, and visibility

Visualization must update in real time and must not block system execution.

---

### Adaptation visualization

The system must display adaptive improvement indicators including:

- stability score trend
- dwell success rate trend
- correction frequency trend
- calibration accuracy improvements
- current correction and calibration state

Visualization must clearly demonstrate system improvement over time.

---

## Code style and architecture guidelines

### Language and framework conventions

- Frontend must use Electron and TypeScript.
- Backend input processing may use Python for computer vision and EEG pipelines.
- Communication between agents and frontend should use structured messaging such as WebSockets.
- System interaction must be isolated in dedicated control agents.
- Configuration must be centralized and profile-aware.

---

### Agent design principles

Agents must follow strict modular design principles:

- Each agent must have a single responsibility.
- Agents must communicate only through defined contracts.
- Agents must be deterministic and testable.
- Agents must fail safely by emitting noop outputs when necessary.
- Agents must emit telemetry for visualization and debugging.

---

## Testing instructions

### Input agent testing

Input agents must be tested for:

- valid gaze tracking behavior
- blink and occlusion handling
- jitter reduction with stabilization
- correct voice command parsing
- valid EEG signal ingestion and processing
- one-euro responsiveness under rapid gaze transitions
- graceful fallback when webcam confidence drops or tracks are lost
- ASR latency and artifact-cleaning mode boundaries for real-time vs non-real-time EEG paths
- deterministic sign-language gesture recognition under occlusion and variable lighting

---

### Control agent testing

Control agents must be tested for:

Cursor control:

- valid cursor movement
- correct clamping and bounds enforcement
- stable behavior during noisy input
- correct dwell activation behavior

Keyboard control:

- correct key press execution
- correct key hold and release behavior
- allowlist enforcement
- rate limit enforcement

---

### Adaptation testing

Adaptation must be tested to ensure:

- calibration reproducibility
- drift detection and correction
- bounded parameter updates
- rollback functionality
- improvement in stability metrics over time
- fail-safe disablement of ErrP/EEG adaptation when quality gates fail
- sign-language fallback and re-activation behavior after gesture dropout

---

## Agent operational constraints

### Latency constraints

The system must operate with real-time responsiveness.

Target latency:

- gaze to cursor update under 50 ms in the primary CV path
- EEG intent generation under 200 ms in baseline mode
- sign-language intent generation under 220 ms in baseline mode
- optional error-correction and learning updates should remain under 150 ms end-to-end where possible
- command routing, parsing, and low-latency intents under 100 ms
- voice command execution under 500 ms (chunked, push-to-talk path)
- visualization refresh rate at least 30 FPS

Known latency exceptions:

- EEG ASR denoising and deep research-mode EEG pipelines are explicitly excluded from the 50 ms gaze control budget.
- If any control path exceeds its threshold for >3 consecutive seconds, the UI must surface degraded-mode status and reduce dependence on that path.

Degraded mode:

- If gaze confidence is below threshold:
  - freeze cursor gain or switch to hold-reduced velocity,
  - disable dwell,
  - display degraded-mode indicator in visualization.
- If voice latency exceeds threshold:
  - reject delayed response as stale,
  - ask for command retry.
- If sign visibility drops:
  - suspend sign output,
  - mark sign modality as degraded until stable reacquisition.

Agents must degrade safely if latency increases.

---

### Reliability constraints

The system must handle failures safely.

Requirements:

- individual agent failures must not crash the system
- invalid inputs must be ignored safely
- only allowlisted actions may reach OS control
- agents must support restart and recovery

---

### Adaptation constraints

Adaptation must never reduce usability or safety.

Requirements:

- adaptation parameters must remain bounded
- unstable adaptation must be rolled back
- profile updates must be versioned and persistent
- adaptation must never introduce unpredictable control behavior
